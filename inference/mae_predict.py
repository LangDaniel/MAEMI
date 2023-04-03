from slices import SliceIterator

import argparse
from pathlib import Path
import sys

import torch
import numpy as np
import h5py
from scipy import ndimage as ndi
import yaml

sys.path.append('./../training/')
from mae3d import models_mae

parser = argparse.ArgumentParser()
parser.add_argument('ckpt', help='checkpoint')
parser.add_argument('data_dir', help='data root dir')
parser.add_argument('label_file', help='lable file')
pargs = parser.parse_args()

ckpt = Path(pargs.ckpt)
data_root_dir = Path(pargs.data_dir)
label_file = Path(pargs.label_file)

trained_dir = ckpt.parent
ckpt = ckpt.name

sys.path.append(str(trained_dir))
import sequence as seq

# choose parameters
# -------------------------------------------------------
stride = (64, 42, 2)
repeat=6
device = 'cuda:0'
filtr = lambda img: ndi.minimum_filter(img, size=(3, 3, 2))
filtr_str = 'min_3x3x2'
# -------------------------------------------------------

# load parameters
class Args:
    def __init__(self, d):
        for key, value in d.items():
            if type(value) == dict:
                value = Args(value)
            setattr(self, key, value)

par_file = Path(trained_dir)/'parameter.yml'
with open (par_file, 'r') as ff:
    par = ff.read()
    args = yaml.safe_load(par)
args['data']['train']['training'] = False
args['data']['general']['augment'] = False
args['data']['general']['pathology'] = 'anomaly'
args['data']['general']['margin'] = [0, 0, 0]
train_data_args = Args({**args['data']['general'], **args['data']['train']})
model_args = Args(args['model'])

model_args.device = device
model_args.input_size = train_data_args.shape

args['data']['train']['label_file'] = label_file
data_args = Args({**args['data']['general'], **args['data']['train']})
data_args.patch_file = data_root_dir / 'patches.h5'

dataset = seq.CustomDataset(data_args)

def uniform_size(data, max_differ=(1, 1, 1), strict=True):
    shape = np.ones(3) * np.inf
    first = True
    for d in data:
        shp = np.minimum(d.shape, shape).astype(int)
        if not first:
            diff = abs(shape - shp)
            if (diff > max_differ).any():
                msg = f'difference {diff} larger than max'
                if strict:
                    raise ValueError(msg)
                print(msg)
        shape = shp
        first = False
    slz = np.s_[:shape[0], :shape[1], :shape[2]]
    out = []
    for d in data:
        out.append(d[slz])
    return out

def get_modality(pid, modality, rtn_info=False):
    data = []
    for mod in modality:
        with h5py.File(dataset.args.patch_file, 'r') as ff:
            lis = ff[pid].attrs['lis']
            imgl = ff[pid][lis][mod][:]
            ris = ff[pid].attrs['ris']
            imgr = ff[pid][ris][mod][:]
        img = np.concatenate([imgl, imgr], axis=1)
        data.append(img)
    data = uniform_size(data, strict=False)
    data = np.array(data)
    if rtn_info:
        return data, [lis, ris]
    return data

dataset.get_modality = get_modality

# load model
model = models_mae.__dict__[model_args.model](
    img_size=model_args.input_size,
    patch_size=model_args.patch_size,
    norm_pix_loss=model_args.norm_pix_loss,
    in_chans=args['model']['in_chans']
)

checkpoint = torch.load(trained_dir / ckpt, map_location='cpu')
msg = model.load_state_dict(checkpoint['model'], strict=False)
print(msg)
_ = model.to(device)

class Predictor():
    def __init__(self, model, dataset, mask_ratio, stride, repeat, filtr):
        self.model = model
        self.dataset = dataset
        self.mask_ratio = mask_ratio
        self.stride = stride
        self.repeat = repeat
        self.filtr = filtr

    @staticmethod
    def zero_pad(img, shape):
         # zero pad if too small
        padw = np.zeros(img.ndim*2)
        for ax in range(img.ndim):
            diff = shape[ax] - img.shape[ax]
            if diff < 0:
                continue
            padw[::2][ax] = int(np.ceil(diff/2)) 
            padw[1::2][ax] = diff//2
        padw = padw.reshape(-1, 2).astype(int)
        if padw.any():
            img = np.pad(img, pad_width=padw)
        return img 
        
    def forward(self, img):
        x = torch.tensor(img)
        x = x.unsqueeze(dim=0)
        x = x.to(device)
        
        loss, y, mask = self.model(x.float(), mask_ratio=self.mask_ratio)
        y = self.model.unpatchify(y)
        y = y.detach().cpu()
    
        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, int(np.prod(self.model.patch_size)))
        mask = model.unpatchify(mask)
        return y.cpu().numpy(), mask.cpu().numpy()
    
    def compute_anomaly(self, img):
        y, _ = self.forward(img)
        diff = (img.squeeze() - y.squeeze())**2
        for ii in range(diff.shape[0]):
            diff[ii] = self.filtr(diff[ii])
            
        return diff
    
    def one_iteration(self, image, patch_size, stride):
        '''
            image: [C, W, H, D]
            patch_size: [W, H, D]
            stride: [W, H, D]
        '''
        diff = np.zeros(image.shape)
        count = np.ones(image.shape)
        
        shape = list(image.shape[-3:])
        shape[1] = shape[1] // 2
        
        # left side
        # --------------------------
        leftiter = SliceIterator(
            shape,
            patch_size,
            stride
        )
        
        for slz in leftiter:
            slz = (slice(None), *slz)
            diff[slz] += self.compute_anomaly(
                image[slz]
            )
            count[slz] += 1
        # -------------------------- 
        # right side
        # --------------------------        
        rightiter = SliceIterator(
            shape,
            patch_size,
            stride,
            [0, shape[1], 0],
        )
        
        for slz in rightiter:
            slz = (slice(None), *slz)
            diff[slz] += self.compute_anomaly(
                image[slz]
            )
            count[slz] += 1
        # -------------------------- 
        diff /= count
        
        return diff, count
    
    def get_anomaly_map(self, pid, modality):
        img, side = self.dataset.get_modality(pid, modality, True)

        # zero pad 
        iz = self.model.img_size
        shape = [self.model.in_chans, iz[0]+1, 2*iz[1]+1, iz[2]+1]
        img = self.zero_pad(img, shape)
        
        diff = np.zeros(img.shape)
        count = np.ones(img.shape)
        for ii in range(self.repeat):
            d, c = self.one_iteration(
                img,
                self.model.img_size,
                self.stride
            )
            diff += d
            count += c
        diff /= count
        
        return diff, count, img, side

pred = Predictor(
    model=model,
    dataset=dataset,
    mask_ratio = args['model']['mask_ratio'],
    stride=stride,
    repeat=repeat,
    filtr=filtr
)

stride_str = 'stride-' + ''.join([f'{ff}x' for ff in stride])[:-1]
info_str = stride_str + f'_repeat-{repeat}_fltr-{filtr_str}'
output_file = ((trained_dir / 'prediction') / info_str) / (label_file.stem + '.h5')
print(output_file)
if output_file.exists():
    raise ValueError('file exists')
output_file.parent.mkdir(parents=True, exist_ok=True)

for pid in dataset.label_df.identifier:
    print(pid)
    diff, count, ref, side = pred.get_anomaly_map(pid, data_args.modalites)
    sgmt, bbox = dataset.get_modality(pid, ['segmentation', 'bbox_roi'])
    
    with h5py.File(output_file, 'a') as ff:
        try:
            ff.create_dataset(
                f'{pid}/diff',
                data=diff
            )
        except (Exception, ArithmeticError) as e:
            raise ValueError(f'{pid} failed: {e}')
print('finished')

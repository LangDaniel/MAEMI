import argparse
from pathlib import Path
import sys

import pandas as pd
import torch
import numpy as np
import h5py
import yaml

from . import models_vit

parser = argparse.ArgumentParser()
parser.add_argument('dir')
parser.add_argument('--ckpt', default='best_loss.pth', type=str)
parser.add_argument('--patch_file', default='', type=str)
parser.add_argument('--label_file', default='', type=str)
parser.add_argument('--dset', default='valid', type=str)
args = parser.parse_args()

sys.path.append(args.dir)
import sequence as seq

class Args:
    def __init__(self, d):
        for key, value in d.items():
            if type(value) == dict:
                value = Args(value)
            setattr(self, key, value)

par_file = Path(args.dir) / 'parameter.yml'
with open (par_file, 'r') as ff:
    par = ff.read()
    par = yaml.safe_load(par)
data_args = Args({**par['data']['general'], **par['data'][args.dset]})
model_args = Args({**par['model'], **par['training']})

model_args.device = 'cpu'
model_args.input_size = data_args.shape

data_args.training = False
data_args.augment = []

# if label or patch file over write default
if args.patch_file:
    data_args.patch_file = Path(args.patch_file)
if args.label_file:
    data_args.label_file = Path(args.label_file)

test_dataset = seq.CustomDataset(data_args)

model = models_vit.__dict__[model_args.model](
    num_classes=model_args.nb_classes,
    drop_path_rate=model_args.drop_path,
    global_pool=model_args.global_pool,
    img_size=model_args.input_size,
    patch_size=model_args.patch_size,
    in_chans=1
)
checkpoint = torch.load(Path(args.dir) / args.ckpt, map_location='cpu')
msg = model.load_state_dict(checkpoint['model'], strict=False)

print(msg)
model.eval()

def forward(img):
    x = torch.tensor(img)
    x = x.unsqueeze(dim=0)

    y = model(x.float())
    y = torch.nn.Softmax(dim=1)(y).detach().numpy()
    return y

columns = ['identifier', 'histology']
results_df = pd.DataFrame(columns=columns+['prediction'])
for idx in range(test_dataset.__len__()):
    X, info = test_dataset.get_case(idx, columns)
    pred = forward(X)[0][1]
    info['prediction'] = pred
    results_df.loc[idx] = info
    idx += 1

results_df['label'] = results_df['histology'].map(
    data_args.label_mapping.__dict__
)

out_file = (Path(args.dir) / 'evaluation' ) / Path(data_args.label_file).name
if out_file.exists():
    raise ValueError('file exists')
out_file.parent.mkdir(exist_ok=True)

results_df.to_csv(out_file, index=False)

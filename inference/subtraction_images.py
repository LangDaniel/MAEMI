from pathlib import Path
import sys

sys.path.append('./../utils/patches/')
from complete_breast_tissue import GetPatches

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy import signal
import SimpleITK as sitk
from sklearn import metrics
import pandas as pd

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

def generate_bbox(size, origin, spacing, bbox_LPS):
    origin = np.array(origin)
    spacing = np.array(spacing)
    bbox_LPS = np.array(bbox_LPS)
    
    sgmt = np.zeros(size)
    
    bbox_CRS = np.zeros(6).astype(int)
    bbox_CRS[::2] = ((bbox_LPS[::2] - origin) / spacing).astype(int)
    bbox_CRS[1::2] = ((bbox_LPS[1::2] - origin) / spacing).astype(int)
    
    bbox_CRS = np.clip(bbox_CRS, a_min=0, a_max=None)
    
    sgmt[
        bbox_CRS[2]:bbox_CRS[3],
        bbox_CRS[0]:bbox_CRS[1],
        bbox_CRS[4]:bbox_CRS[5]
    ] = 1
    return sgmt

label_file = Path('./../data/labels/anomaly_no_bilateral/test.csv')
label_df = pd.read_csv(label_file)

bbox_df = pd.read_csv('./../data/bboxes/bboxes_GTV_LPS.csv')
bbox_df = bbox_df.set_index('identifier')

tissue_bbox_file = './../data/bboxes/adjusted_size_200-350x200-350x50-100/breast_tissue_complete_from_sgmt.csv'
path_file = './../data/file_paths.csv'
side_file = './../data/TCIA/Duke-Breast-Cancer-MRI/Clinical_and_Other_Features.xlsx'
output_file = ''
rescale = False
modalities=['t1_FS'] + [f'phase{ii}' for ii in range(1,5)]
normalize=modalities
bias_correction = []
crop = 'before'
std = 0.25
mean = 0.5


spacing = (0.75, 0.75, 1.0)
min_size_mm = [False]*3

patch_gen = GetPatches(
    bbox_file=tissue_bbox_file,
    path_file=path_file,
    side_file=side_file,
    output_file=output_file,
    crop=crop,
    bias_correction=bias_correction,
    rescale=rescale,
    normalize=normalize,
    spacing=spacing,
    min_size=min_size_mm,
    modalities=modalities,
)

def diff_map(img1, img2):
    return (img1 - img2)**2

def normalize_image(data, std=1., mean=3.):
    f = sitk.NormalizeImageFilter()
    data = f.Execute(data)
    return (std*data)+mean

patch_gen.normalize_image = lambda img: normalize_image(img, std=std, mean=mean)

root_dir = Path('./../data/TCIA/Duke-Breast-Cancer-MRI/')
outfile = Path(f'./prediction/contrast_maps/{label_file.stem}.h5')
if outfile.exists():
    raise ValueError('file exists')
outfile.parent.mkdir(parents=True, exist_ok=True)

reference = 't1_FS'
comparison = 't1_non_FS'

for pid in label_df.identifier.values:
    print(pid)
    case_dir = root_dir / pid
    contrast = [ff.stem for ff in case_dir.iterdir() if ff.stem.startswith('phase')]
    
    sgmt = patch_gen.get_case(pid, 'segmentation')
    #sgmt = (sgmt > 0.5).astype(int)
    #sgmt = ndi.grey_erosion(sgmt, size=(8, 8, 4))
    
    ref, roi_bbox = patch_gen.get_case(pid, reference, rtn_bbox=True)
    comp = patch_gen.get_case(pid, comparison)
    post = []
    for con in contrast:
        try:
            patch = patch_gen.get_case(pid, con)
            post.append(patch)
        except:
            print(f'failed for {con}')
    ref, comp, *post = uniform_size([ref, comp, *post], strict=False)
    post = np.array(post)
    
    diff = np.zeros(ref.shape)
    for p in post:
        diff += diff_map(ref, p)
    diff /= len(post)
    
    origin = roi_bbox[::2]
    bbox = generate_bbox(ref.shape, origin, spacing, bbox_df.loc[pid].values).astype(int)
    with h5py.File(outfile, 'a') as ff:
        ff.create_dataset(
            f'{pid}/diff',
            data=diff
        )
        ff.create_dataset(
            f'{pid}/bbox_roi',
            data=bbox
        )
        ff.create_dataset(
            f'{pid}/segmentation',
            data=sgmt
        )
        ff.create_dataset(
            f'{pid}/reference',
            data=ref
        )
        ff.create_dataset(
            f'{pid}/comparison',
            data=comp
        )


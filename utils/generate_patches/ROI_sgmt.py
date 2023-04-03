from pathlib import Path
import pickle

import h5py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

root_dir = Path('./../../data/patch_data/complete_clc_mean05_std025_space075x075x10_size200x200x50')

with open(root_dir / 'patches.pkl', 'rb') as ff:
    spacing = pickle.load(ff)['spacing']

origin_df = pd.read_csv(root_dir /'patches.csv')[['identifier', 'L_min', 'P_min', 'S_min']]
origin_df = origin_df.drop_duplicates()
origin_df = origin_df.set_index('identifier')

bbox_df = pd.read_csv('./../../data/bboxes/bboxes_GTV_LPS.csv')
bbox_df = bbox_df.set_index('identifier')

def get_bbox_sgmt(size, origin, spacing, bbox_LPS):
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

identifier = []
with h5py.File(root_dir / 'patches.h5', 'r') as ff:
    for pid in list(ff.keys()):
        if 'bbox_roi' not in list(ff[pid]['normal'].keys()):
            identifier.append(pid)

for pid in identifier:
    print(pid)
    try:
        with h5py.File(root_dir / 'patches.h5', 'r') as ff:
            lis = ff[pid].attrs['lis']
            imgl = ff[pid][lis]['t1_FS'][:]
            ris = ff[pid].attrs['ris']
            imgr = ff[pid][ris]['t1_FS'][:]
        img = np.concatenate([imgl, imgr], axis=1)
        
        origin = origin_df.loc[pid].values
        bbox_LPS = bbox_df.loc[pid].values
        sgmt = get_bbox_sgmt(img.shape, origin, spacing, bbox_LPS)
        
        width = sgmt.shape[1]

        with h5py.File(root_dir /'patches.h5', 'a') as ff:
            ff.create_dataset(
                f'{pid}/{lis}/bbox_roi',
                data=sgmt[:, :width//2, :]
            )
            ff.create_dataset(
                f'{pid}/{ris}/bbox_roi',
                data=sgmt[:, width//2:, :]
            )
    except (Exception, ArithmeticError) as e:
        print(f'failed {e}')

print('Done')

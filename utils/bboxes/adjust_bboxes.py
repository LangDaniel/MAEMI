from pathlib import Path
import pandas as pd
import numpy as np

bb_file = Path('./../../data/bboxes/breast_tissue_complete_from_sgmt.csv')
bbox_df = pd.read_csv(bb_file)

def adjust_to_max_size(ax_min, ax_max, max_size, mode):
    ax_size = ax_max - ax_min
    diff = ax_size - max_size
    if diff < 0:
        return ax_min, ax_max
    
    if mode == 'left':
        target_min = ax_min
    elif mode == 'right':
        target_min = ax_max - max_size -1
    elif mode == 'center':
        target_min = ax_min + diff/2
    else:
        raise ValueErro('unknown mode')
    target_max = target_min + max_size
    return target_min, target_max

def adjust_to_min_size(ax_min, ax_max, min_size, mode):
    ax_size = ax_max - ax_min
    diff = min_size - ax_size
    if diff < 0:
        return ax_min, ax_max
    
    if mode == 'left':
        target_min = ax_min
    elif mode == 'right':
        target_min = ax_max - max_size -1
    elif mode == 'center':
        target_min = ax_min - diff/2
    else:
        raise ValueErro('unknown mode')
    target_max = target_min + min_size
    return target_min, target_max

min_size = [200, 200, 50]
max_size = [350, 350, 100]

mode = ['center', 'left', 'center']

adjusted_df = pd.DataFrame(columns=list(bbox_df))
idx = 0

columns = ['L_min', 'L_max', 'P_min', 'P_max', 'S_min', 'S_max']
for _, row in bbox_df.iloc[:].iterrows():
    bbox = row[columns].values
    min_bb = bbox[::2]
    max_bb = bbox[1::2]
    adjusted_bb = []
    for ii, ax in enumerate('LPS'):
        ax_min = row[f'{ax}_min']
        ax_max = row[f'{ax}_max']
        # adjust to max
        ax_min, ax_max = adjust_to_max_size(
            ax_min,
            ax_max,
            max_size[ii],
            mode[ii]
        )
        # adjust to min
        ax_min, ax_max = adjust_to_min_size(
            ax_min,
            ax_max,
            min_size[ii],
            mode[ii]
        )
        adjusted_bb += [ax_min, ax_max]
    adjusted_df.loc[idx] = [row['identifier'], *adjusted_bb]
    idx += 1

gtv_df = pd.read_csv('./../../data/bboxes/bboxes_GTV_LPS.csv')
merged_df = adjusted_df.merge(gtv_df, on='identifier', suffixes=('_adj', '_gtv'))

while(True):
    fin = True
    print('iterating')
    for idx, row in merged_df.iloc[:].iterrows():
        pid = row['identifier']
        for ax in 'LPS':
            if row[f'{ax}_min_gtv'] < row[f'{ax}_min_adj']:
                #print(f'{pid} - {ax} - min')
                fin = False
                val = float(row[[f'{ax}_min_gtv']])
                merged_df.loc[idx, f'{ax}_min_adj'] = val - 20
            if row[f'{ax}_max_gtv'] > row[f'{ax}_max_adj']:
                #print(f'{pid} - {ax} - max')
                fin = False
                val = float(row[[f'{ax}_max_gtv']])
                merged_df.loc[idx, f'{ax}_max_adj'] = val + 20
    if fin:
        break

df = merged_df[list(merged_df)[:7]].copy()
df = df.rename(
    columns={key: value for key, value in zip(list(df), list(adjusted_df))}
)

mima_str = 'x'.join([f'{mi}-{ma}' for mi, ma in zip(min_size, max_size)])

to_disk = True
if to_disk:
    out_file = (bb_file.parent / f'adjusted_size_{mima_str}') / bb_file.name
    out_file.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(out_file, index=False)

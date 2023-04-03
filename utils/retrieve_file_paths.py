from pathlib import Path
import pandas as pd
import numpy as np

root_dir = Path('./../data/TCIA/Duke-Breast-Cancer-MRI/')
cases = sorted([ff for ff in root_dir.iterdir() if ff.is_dir()])

sgmt_df = pd.DataFrame(columns=['identifier', 'segmentation'])
idx = 0

for ff in cases:
    pid = ff.stem
    file = ff / f'SEG/{pid}.nii.gz'
    if file.exists():
        sgmt_df.loc[idx] = [pid, str(file)]
        idx += 1
    else:
        print(f'missing {pid}')

cols = ['identifier', 't1_FS', 't1_non_FS', 'phase1', 'phase2', 'phase3', 'phase4']
img_df = pd.DataFrame(columns=cols)

idx = 0
for ff in cases:
    pid = ff.stem
    file_path = []
    for sub in cols[1:]:
        path = ff / sub
        if path.exists():
            file_path.append(path)
        else:
            file_path.append('None')
    
    img_df.loc[idx] = [pid] + file_path
    idx += 1

df = img_df.merge(sgmt_df, on='identifier', how='outer')

for col in list(df)[1:]:
    count = np.count_nonzero(df[col] == 'None')
    print(f'{col} missing {count} instances')

mask = (
    (df['t1_FS'] != 'None') &\
    (df['t1_non_FS'] != 'None') &\
    (df['phase1'] != 'None') &\
    (df['phase2'] != 'None')
)

df = df[mask]
df.to_csv('./../data/file_paths.csv', index=False)

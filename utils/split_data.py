from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

size_df = pd.read_csv('./../data/bboxes/bboxes_GTV_LPS.csv')

clinical_df = pd.read_excel(
    './../data/TCIA/Duke-Breast-Cancer-MRI/Clinical_and_Other_Features.xlsx',
    skiprows=1
).iloc[1:]
clinical_df = clinical_df.rename(columns={'Patient ID': 'identifier'})

df = size_df.merge(clinical_df, on='identifier', how='outer').fillna(-1)

# remove bilateral cases
df = df[df['Bilateral Information'] != 1]
df = df.reset_index(drop=True)

for ax in 'LPS':
    df[f'{ax}_size'] = abs(df[f'{ax}_max'] - df[f'{ax}_min'])

df['volume'] = df['L_size'] * df['P_size'] * df['S_size'] / (10**3)

def round_to_base(value, base, maximum):
    value = min(value, maximum)
    return round(value/base)*base

volume = df['volume'].apply(lambda x: round_to_base(x, 10, 700)).astype(int)

manufacturer = df['Manufacturer'].astype(int).fillna(-1)
tesla = df['Field Strength (Tesla)'].astype(int).fillna(-1)

split_values = 100000 * volume + 10 * manufacturer + tesla

cut_split = {}
for key, value in split_values.value_counts().iteritems():
    if value < 10:
        cut_split[key] = 0
split_values = split_values.replace(cut_split)

split_size = 100
split_gen = StratifiedShuffleSplit(n_splits=1, test_size=split_size, random_state=42)

for training_idx, test_idx in split_gen.split(df, split_values):
    training_df = df.iloc[training_idx].copy().reset_index(drop=True)
    split_values = split_values[training_idx]
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)
    
split_size = 50
split_gen = StratifiedShuffleSplit(n_splits=1, test_size=split_size, random_state=42)

for train_idx, valid_idx in split_gen.split(training_df, split_values):
    train_df = training_df.iloc[train_idx].copy().reset_index(drop=True)
    valid_df = training_df.iloc[valid_idx].copy().reset_index(drop=True)

columns = ['identifier', 'volume']
train_df = train_df[columns]
valid_df = valid_df[columns]
test_df = test_df[columns]

to_disk = False
root_dir = Path('./../data/labels/anomaly_no_bilateral/')


if to_disk:
    root_dir.mkdir(parents=True)
    train_df.to_csv(root_dir / 'train.csv', index=False)
    valid_df.to_csv(root_dir / 'valid.csv', index=False)
    test_df.to_csv(root_dir / 'test.csv', index=False)

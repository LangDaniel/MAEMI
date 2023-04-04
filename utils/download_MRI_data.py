from pathlib import Path
import requests
from zipfile import ZipFile
import pandas as pd
import numpy as np

class TCIAClient:
    def __init__(self, url):
        if url[-1] != '/':
            url = url + '/'
        self.base_url = url
        
    def get_json(self, get, params={}):
        params['format'] = 'json'
        url = self.base_url + get

        return requests.get(url, params=params).json()
        
    def get_image(self,
                  series_instance_uid,
                  local_path,
                  unzip,
                  remove_zip):
        
        url = self.base_url + 'getImage'
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True)
        
        params = {'SeriesInstanceUID': series_instance_uid}
        with requests.get(url, params=params, stream=True) as req:
            req.raise_for_status()
            with open(local_path, 'wb') as ff:
                for chunk in req.iter_content(chunk_size=8192): 
                    ff.write(chunk)
        if unzip:
            with ZipFile(local_path, 'r') as zz:
                try:
                    zz.extractall(local_path.parent)
                except:
                    print(f'unzipping for {local_path} failed')
            if remove_zip:
                local_path.unlink()
        return True

base_url = 'https://services.cancerimagingarchive.net/services/v4/TCIA/query'
client = TCIAClient(base_url)

collection = 'Duke-Breast-Cancer-MRI'
series_df = pd.DataFrame(
    data=client.get_json('getSeries', {'Collection': collection})
)
patient_df = pd.DataFrame(
    data=client.get_json('getPatientStudy', {'Collection': collection})
)
df = series_df.merge(patient_df)

# generate dict for unique naming of sequences

modality_dict = {}

t1_FS_names = [
    'Ax Vibrant MultiPhase',
    'ax 3d dyn',
    'ax 3d dyn MP',
    'ax 3d dyn pre',
    'ax 3d pre',
    'ax dyn',
    'ax dyn pre',
    'ax dynamic',
    't1_fl3d_tra_dynVIEWS_spair_ pre'
]
for key in t1_FS_names:
    modality_dict[key] = 't1_FS'

t1_non_FS_names = [
    'AX IDEAL Breast',
    'ax 3d t1 bilateral',
    'ax t1',
    'ax t1 +c',
    'ax t1 2mm',
    'ax t1 3mm',
    'ax t1 pre',
    'ax t1 repeat',
    'ax t1 tse',
    'ax t1 tse +c',
    'Ax 3D T1 NON FS',
    'ax 3d t1 bilateral'
]
for key in t1_non_FS_names:
    modality_dict[key] = 't1_non_FS'

first_seq_names = [
    '1st ax dyn',
    'Ph1/Ax Vibrant MultiPhase',
    'Ph1/ax 3d dyn',
    'Ph1/ax 3d dyn MP',
    'Ph1/ax dyn',
    'Ph1/ax dynamic',
    'ax 3d dyn 1st pass',
    'ax dyn 1st pas',
    'ax dyn 1st pass',
    't1_fl3d_tra_dynVIEWS_spair 1st pass'
]
for key in first_seq_names:
    modality_dict[key] = 'phase1'

second_seq_names = [
    '2nd ax dyn',
    'Ph2/Ax Vibrant MultiPhase',
    'Ph2/ax 3d dyn',
    'Ph2/ax 3d dyn MP',
    'Ph2/ax dyn',
    'Ph2/ax dynamic',
    'ax 3d dyn 2nd pass',
    'ax dyn 2nd pass',
    't1_fl3d_tra_dynVIEWS_spair_2nd pass'
]
for key in second_seq_names:
    modality_dict[key] = 'phase2'

third_seq_names = [
    '3rd ax dyn',
    'Ph3/Ax Vibrant MultiPhase',
    'Ph3/ax 3d dyn',
    'Ph3/ax 3d dyn MP',
    'Ph3/ax dyn',
    'Ph3/ax dynamic',
    'ax 3d dyn 3rd pass',
    'ax dyn 3rd pass',
    't1_fl3d_tra_dynVIEWS_spair_3rd pass'
]
for key in third_seq_names:
    modality_dict[key] = 'phase3'

fourth_seq_names = [
    '4th ax dyn',
    'Ph4/Ax Vibrant MultiPhase',
    'Ph4/ax 3d dyn',
    'Ph4/ax 3d dyn MP',
    'Ph4/ax dyn',
    'Ph4/ax dynamic',
    'ax 3d dyn 4th pass',
    'ax dyn 4th pass',
    't1_fl3d_tra_dynVIEWS_spair_4th pass'
]
for key in fourth_seq_names:
    modality_dict[key] = 'phase4'

df['Type'] = df['SeriesDescription'].map(modality_dict)

# remove segmentations from df
df = df[df['Modality'] == 'MR']

root_dir = Path('./../data/TCIA') / collection

for ii, row in df.iloc[:20].iterrows():
    pid = row['PatientID']
    typ = row['Type']
    print(f'{pid}: {typ}')
    uid = row['SeriesInstanceUID']
    path = root_dir / (row['PatientID'] + f'/{typ}/{typ}.zip')
    if path.exists():
        print(f'exists: {path}')
        continue
    try:
        client.get_image(uid, path, True, True)
    except Exception as e:
        print(f'failed {e}')

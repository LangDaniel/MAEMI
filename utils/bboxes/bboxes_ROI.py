# run in the medimg conda environment
from pathlib import Path
import SimpleITK as sitk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

root_dir = Path('./../../data/TCIA/Duke-Breast-Cancer-MRI/')
bbox_RCS_df = pd.read_excel(root_dir.parent / 'Annotation_Boxes.xlsx')
bbox_RCS_df = bbox_RCS_df.set_index('Patient ID')
verbose = False

#-------------------------------------------------------------------- 
# Duke dataset specific stuff, the order of coorinates is strange, see:
# https://sites.duke.edu/mazurowski/resources/breast-cancer-mri-dataset/
# -------------------------------------------
def get_slice_reader(path):
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    return reader

def get_pos_info(path):
    '''returns the z coordinate and slice number of the image'''
    reader = get_slice_reader(path)
    instance_n = int(reader.GetMetaData('0020|0013'))
    z_pos = reader.GetOrigin()[-1]
    return z_pos, instance_n

def is_ordered(folder):
    folder = Path(folder)
    # take two random files
    files = [ff for ff in folder.iterdir()][:2]

    z_pos = []
    number = []

    for ff in files:
        zp, numb = get_pos_info(str(ff))
        z_pos.append(zp)
        number.append(numb)

    # sort pos after slice number
    if number[0] > number[1]:
        z_pos[0], z_pos[1] = z_pos[1], z_pos[0]

    # check order
    if z_pos[0] < z_pos[1]:
        return True
    return False

def switch_slices(init, fin, size):
    ii = size - fin + 1
    ff = size - init + 1
    return ii, ff

def get_itk_from_path(path, orientation=False):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(path))
    reader.SetFileNames(dicom_names)

    data = reader.Execute()
    if orientation:
        data = sitk.DICOMOrient(data, orientation)

    return data

def get_bbox_LPS(dcm_folder, bbox_RCS):
    ds = get_itk_from_path(dcm_folder)
    
    # change slice order
    # (Duke dataset specific stuff)
    if not is_ordered(dcm_folder):
        shape = ds.GetSize()
        bbox_RCS[-2:] = switch_slices(bbox_RCS[-2], bbox_RCS[-1], shape[-1])
        
    bbox_CRS = bbox_RCS.copy()
    bbox_CRS[:2] = bbox_RCS[2:4]
    bbox_CRS[2:4] = bbox_RCS[:2]
    
    bbox_LPS = np.zeros(6)
    bbox_LPS[::2] = ds.TransformIndexToPhysicalPoint(
        bbox_CRS[::2].tolist()
    )
    bbox_LPS[1::2] = ds.TransformIndexToPhysicalPoint(
        bbox_CRS[1::2].tolist()
    )

    bbox_LPS[:2] = np.sort(bbox_LPS[:2])
    bbox_LPS[2:4] = np.sort(bbox_LPS[2:4])
    bbox_LPS[4:6] = np.sort(bbox_LPS[4:6])

    return bbox_LPS

columns = ['identifier', 'L_min', 'L_max', 'P_min', 'P_max', 'S_min', 'S_max']
bbox_LPS_df = pd.DataFrame(columns=columns)

idx = 0
for pid, bb in bbox_RCS_df.iterrows():
    print(pid)
    bbox_RCS = bb.values
    path = (root_dir / pid) / 't1_FS'

    try:
        bbox_LPS = get_bbox_LPS(path, bbox_RCS)
        bbox_LPS_df.loc[idx] = [pid, *bbox_LPS]
        idx +=1
    except (Exception, ArithmeticError) as e: 
        if verbose:
            print(f'failed: {e}')
        else:
            print('failed')

bbox_LPS_df.to_csv('./../../data/bboxes/bboxes_ROI_LPS.csv', index=False)

import argparse
import os
from pathlib import Path

import SimpleITK as sitk
import numpy as np

def get_itk_from_path(path, orientation='LPS'):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(path))
    reader.SetFileNames(dicom_names)

    data = reader.Execute()
    if orientation:
        data = sitk.DICOMOrient(data, orientation)

    return data

def get_array_from_itk(image):
    data = sitk.GetArrayFromImage(image)
    data = np.moveaxis(data, 0, -1) 
    return data

def zscore_image(image_array):
    """
    Convert intensity values in an image to zscores:
    zscore = (intensity_value - mean) / standard_deviation

    Parameters
    ----------
    image_array: np.array
        3D numpy array constructed from dicom files
    Returns
    -------
    np.array
        Image with zscores for values

    """

    image_array = (image_array - np.mean(image_array)) / np.std(image_array)

    return image_array

parser = argparse.ArgumentParser()
parser.add_argument('dcm_folder', help='path to dicom dir')
parser.add_argument('model', help='path to predict.py file')
parser.add_argument('sgmt_folder', help='path to output file')

args = parser.parse_args()

dcm_folder = Path(args.dcm_folder)
pid = dcm_folder.parent.stem

tmp_folder = Path('./.tmp/')
file_name = pid + '.npy'

tmp_img_folder = tmp_folder / 'image'
tmp_img_folder.mkdir(parents=True)
tmp_mask_folder = tmp_folder / 'mask'
tmp_mask_folder.mkdir(parents=True)

sgmt_file = Path(args.sgmt_folder) / (pid + '.nii.gz')
sgmt_file.parent.mkdir(parents=True)

model = Path(args.model)
weights = Path(model).parent / 'trained_models/breast_model.pth'

img_ds = get_itk_from_path(dcm_folder)
img = get_array_from_itk(img_ds)

with open(tmp_img_folder / file_name, 'wb') as f:
    np.save(f, zscore_image(img))

os.system(
    f'python {model} '\
    f'--target-tissue breast '\
    f'--image {tmp_img_folder} '\
    f'--save-masks-dir {tmp_mask_folder} '\
    f'--model-save-path {weights}'
)

with open(tmp_mask_folder / file_name, 'rb') as f:
    pred = np.load(f)

pred = np.moveaxis(pred, -1, 0)
sgmt_ds = sitk.GetImageFromArray(pred.astype('float32'))
sgmt_ds.SetOrigin(img_ds.GetOrigin())
sgmt_ds.SetSpacing(img_ds.GetSpacing())
sgmt_ds.SetDirection(img_ds.GetDirection())
sitk.WriteImage(sgmt_ds, sgmt_file)

os.system(f'rm -rf {tmp_folder}')

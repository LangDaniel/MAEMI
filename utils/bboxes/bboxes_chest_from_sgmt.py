# run in the medimg environment
from pathlib import Path

import pandas as pd
import numpy as np
import SimpleITK as sitk

class BboxGen():
    def __init__(self, root_dir, sgmt_threshold = 0.5):
        self. cases = sorted([
            ff for ff in Path(root_dir).iterdir() if ff.is_dir()
        ])
        self.sgmt_th = sgmt_threshold
        self.columns = [
            'identifier',
            'L_min',
            'L_max',
            'P_min',
            'P_max',
            'S_min',
            'S_max'
        ]
    
    @staticmethod
    def read_dcm_folder(path, orientation='LPS'):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(path))
        reader.SetFileNames(dicom_names)
    
        data = reader.Execute()
        if orientation:
            data = sitk.DICOMOrient(data, orientation)
    
        return data
    
    @staticmethod
    def read_nii_file(path, orientation='LPS'):
        data = sitk.ReadImage(path)
        data = sitk.DICOMOrient(data, orientation)
    
        return data
    
    @staticmethod
    def get_array_from_itk(image):
        data = sitk.GetArrayFromImage(image)
        data = np.moveaxis(data, 0, -1) 
        return data
    
    @staticmethod
    def get_bbox_CRS(sgmt):
        bbox_RCS = np.zeros(6)
        contour = np.where(sgmt)
        bbox_RCS[::2] = np.min(contour, axis=1)
        bbox_RCS[1::2] = np.max(contour, axis=1)
        
        # RCS to CRS
        bbox_CRS = bbox_RCS.copy()
        bbox_CRS[0:2] = bbox_RCS[2:4]
        bbox_CRS[2:4] = bbox_RCS[0:2]
        return bbox_CRS.astype(int)

    def get_bbox_LPS(
        self,
        case_dir,
        ):
    
        pid = case_dir.stem
        
        # get image as itk
        img_file = case_dir / 't1_FS'
        img_ds = self.read_dcm_folder(str(img_file))
        
        # get sgmt np array
        sgmt_file = case_dir / ('SEG/' + pid + '.nii.gz')
        sgmt_ds = self.read_nii_file(str(sgmt_file))
        sgmt = self.get_array_from_itk(sgmt_ds)
        sgmt = (sgmt > self.sgmt_th).astype(int)
        
        bbox_CRS = self.get_bbox_CRS(sgmt)
    
        bbox_LPS = np.zeros(6)
        bbox_LPS[::2] = img_ds.TransformIndexToPhysicalPoint(
            list(bbox_CRS[::2].astype('int').tolist())
        )
        bbox_LPS[1::2] = img_ds.TransformIndexToPhysicalPoint(
            list(bbox_CRS[1::2].astype('int').tolist())
        )
    
        return bbox_LPS
    
    def generate_bboxes(self, save_to='', folder=True):
        if save_to:
            save_to = Path(save_to)
            if not save_to.exists():
                if folder and not save_to.parent.exists():
                    save_to.parent.mkdir(
                        parents=True
                    )
            else:
                raise ValueError('file exists')
        
        bbox_df = pd.DataFrame(columns=self.columns)
        idx = 0
        
        for case in self.cases:
            pid = case.stem
            print(f'{pid}')
            try:
                bb = self.get_bbox_LPS(case)
                bbox_df.loc[idx] = [pid, *bb]
                idx +=1
            except Exception as e:
                print(f'failed:{e}')
            if save_to:
                bbox_df.to_csv(save_to, index=False)
        return bbox_df

if __name__ == '__main__':
    root_dir = Path('./../../data/TCIA/Duke-Breast-Cancer-MRI/')
    bbgen = BboxGen(root_dir)
    bbgen.generate_bboxes('./../../data/bboxes/breast_tissue_complete_from_sgmt.csv')

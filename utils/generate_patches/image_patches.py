from pathlib import Path
import pickle
import SimpleITK as sitk
import numpy as np
import pandas as pd
import h5py

class GetPatches():
    
    def __init__(
        self,
        bbox_file,
        path_file,
        side_file,
        output_file,
        spacing,
        min_size,
        crop='before',
        bias_correction=[],
        rescale=False,
        normalize=[],
        bbox_cols=[
            'L_min',
            'L_max',
            'P_min',
            'P_max',
            'S_min',
            'S_max',
        ],  
        modalities=[
            't1_non_FS',
            't1_FS',
            'segmentation',
        ]
    ):

        # merge tissue bbox file with path file
        # in the case_df data frame
        # -------------------------------------------------
        tissue_bbox_df = pd.read_csv(bbox_file)
        self.bbox_cols = bbox_cols
        path_df = pd.read_csv(path_file)
        case_df = tissue_bbox_df.merge(
            path_df,
            on='identifier',
            how='inner',
        ) 
        self.case_df = case_df.set_index('identifier')
        # -------------------------------------------------

        # read file holding info about the side
        # of the respective pathology
        # -------------------------------------------------
        side_df = pd.read_excel(
            side_file,
            skiprows=1
        ).iloc[1:]
        side_df['Side'] = side_df['Position'].str[:1]
        self.side_df = side_df.set_index('Patient ID')
        # -------------------------------------------------

        self.output_file = Path(output_file)

        self.spacing = np.array(spacing)
        self.min_size = min_size
        
        # crop patch before or after intensity value manipulations
        assert crop == 'before' or crop == 'after'
        self.crop = crop

        self.bias_correction = bias_correction
        self.rescale = rescale
        self.normalize = normalize

        self.modalities = modalities

        for bc in bias_correction:
            if bc not in modalities:
                raise ValueError(f'bias correction {bc} not in modalities: {modalities}')
        for norm in normalize:
            if norm not in modalities:
                raise ValueError(f'normalize {norm} not in modalities: {modalities}')

        if not rescale and not normalize:
            print('no rescaling and no normalization used!')
        if rescale and normalize:
            print('!!!! both and normalization used !!!!')

    ######################## IO stuff ##########################

    def get_itk_from_path(self, path, orientation='LPS'):
        path = Path(path)
        if path.is_dir():
            return self.read_dcm_folder(path, orientation) 
        if (path.suffix == '.gz') or (path.suffix == '.nii'): 
            return self.read_nii_file(path, orientation)
        
        raise ValueError('unknown file format')

    @staticmethod
    def read_dcm_folder(path, orientation):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(path))
        reader.SetFileNames(dicom_names)

        data = reader.Execute()
        data = sitk.DICOMOrient(data, orientation)

        return data

    @staticmethod
    def read_nii_file(path, orientation='LPS'):
        data = sitk.ReadImage(str(path))
        data = sitk.DICOMOrient(data, orientation)
    
        return data

    @staticmethod
    def get_array_from_itk(image):
        data = sitk.GetArrayFromImage(image)
        return np.moveaxis(data, 0, -1)
        
    #################### read bbox #############################

    def get_bbox_LPS(self, pid):
        row = self.case_df.loc[pid]
        bbox = row[self.bbox_cols].values

        if self.min_size:
            bbox = self.fit_bbox(bbox)

        return bbox

    def fit_bbox(self, bbox):
        for jj in range(0, 3): 
            diff = bbox[1::2][jj] - bbox[::2][jj]
            margin = self.min_size[jj] - diff
            margin = np.clip(margin, a_min=0, a_max=None) 

            bbox[::2][jj] -= margin / 2 
            bbox[1::2][jj] += margin / 2 

        return bbox

    def get_side_split(self, pid, pathology):
        assert pathology in ['normal', 'anomaly']
        side = self.side_df.loc[pid]['Side']
        if pathology == 'normal':
            if side == 'R':
                return lambda x: [x//2, x]
            if side == 'L':
                return lambda x: [0, x//2]
        if pathology == 'anomaly':
            if side == 'L':
                return lambda x: [x//2, x]
            if side == 'R':
                return lambda x: [0, x//2]
        return False

    ######################### crop ##############################

    def resample_img(self, itk_image, is_label=False):

        original_spacing = itk_image.GetSpacing()
        original_size = itk_image.GetSize()

        spacing = np.zeros(3)
        for ii in range(0, 3):
            if self.spacing[ii] == False:
                spacing[ii] = original_spacing[ii]
            else:
                spacing[ii] = self.spacing[ii]

        out_size = [
            int(np.round(original_size[0] * (original_spacing[0] / spacing[0]))),
            int(np.round(original_size[1] * (original_spacing[1] / spacing[1]))),
            int(np.round(original_size[2] * (original_spacing[2] / spacing[2])))]

        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(spacing)
        resample.SetSize(out_size)
        resample.SetOutputDirection(itk_image.GetDirection())
        resample.SetOutputOrigin(itk_image.GetOrigin())
        resample.SetTransform(sitk.Transform())
        if is_label:
            resample.SetDefaultPixelValue(0)
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
            resample.SetInterpolator(sitk.sitkBSpline)

        return resample.Execute(itk_image)

    def zero_pad(self, itk_img, bbox, const=0):
        size = np.array(itk_img.GetSize())
        spacing = np.array(itk_img.GetSpacing())
    
        img_lower = itk_img.GetOrigin()
        img_upper =  img_lower + size * spacing
    
        bb_lower = bbox[::2]
        bb_upper = bbox[1::2]
    
        lower_pad = np.zeros(3)
        upper_pad = np.zeros(3)
        for ii in range(0, 3): 
            lower_diff = img_lower[ii] - bb_lower[ii]
            if lower_diff > 0:
                lower_pad[ii] = np.ceil(lower_diff / spacing[ii]).astype(int)
    
            upper_diff = bb_upper[ii] - img_upper[ii]
            if upper_diff > 0:
                upper_pad[ii] = (np.ceil(upper_diff / spacing[ii])).astype(int)

        if not lower_pad.any() and not upper_pad.any():
            return itk_img
        print('zero padding')
        # convert to list due to sitk bug
        lower_pad = lower_pad.astype('int').tolist()
        upper_pad = upper_pad.astype('int').tolist()

        return sitk.ConstantPad(itk_img, lower_pad, upper_pad, constant=const)

    @staticmethod
    def bias_field_correction(image):
        mask = sitk.OtsuThreshold(image, 0, 1, 200)
        mask = sitk.Cast(mask, sitk.sitkUInt8)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        numberFittingLevels = 4
        
        corrected_image = corrector.Execute(
            sitk.Cast(image, sitk.sitkFloat32),
            mask,
        )
        return corrected_image

    @staticmethod
    def rescale_linear(data, scaling):
        [in_low, in_high], [out_low, out_high] = scaling

        m = (out_high - out_low) / (in_high - in_low)
        b = out_low - m * in_low
        data = m * data + b
        data = np.clip(data, out_low, out_high)
        return data

    @staticmethod
    def normalize_image(img):
        raise NotImplementedError('normalization function not implemented')

    @staticmethod
    def crop_patch(data, bbox):
        xii, xff, yii, yff, zii, zff = bbox.astype(int)
        return data[xii:xff, yii:yff, zii:zff]
    
    def get_bbox_RCS(self, ds, bbox_LPS):
        bbox_RCS = np.zeros(6) 
        bbox_RCS[::2] = ds.TransformPhysicalPointToIndex(
            bbox_LPS[::2]
        )
        bbox_RCS[1::2] = ds.TransformPhysicalPointToIndex(
            bbox_LPS[1::2]
        )
        return bbox_RCS.astype(int)

    def get_case(self, pid, modality, rtn_bbox=False):
        bbox_LPS = self.get_bbox_LPS(pid)
        path = self.case_df.loc[pid][modality]
        
        try:
            # read and reample image
            ds = self.get_itk_from_path(path)
            if self.spacing.any():
                ds = self.resample_img(ds, is_label=False)
            ds = self.zero_pad(ds, bbox_LPS)
    
            # get bbox
            bbox_RCS = self.get_bbox_RCS(ds, bbox_LPS)

            if self.crop == 'before':
                ds = self.crop_patch(ds, bbox_RCS)

            # intensity modifications
            if modality in self.bias_correction:
                print('bias correction')
                ds = self.bias_field_correction(ds)
            if modality in self.normalize:
                print('normalization')
                ds = self.normalize_image(ds)

            if self.crop == 'after':
                ds = self.crop_patch(ds, bbox_RCS)

            img = self.get_array_from_itk(ds)
            if self.rescale:
                img = self.rescale_linear(img, self.rescale)
        except (Exception, ArithmeticError) as e: 
            raise ValueError(f'{pid} failed: {e}')

        if rtn_bbox:
            return img, bbox_LPS
        return img

    def pathology_of_side(self, pid, side):
        assert side in ['L', 'R']
        anomaly_side = self.side_df.loc[pid]['Side']
        if side == anomaly_side:
            return 'anomaly'
        return 'normal'

    def to_disk(self):
        if self.output_file.exists():
            raise ValueError('output file exists')
        if not self.output_file.parent.exists():
            self.output_file.parent.mkdir(parents=True)

        disk_df = pd.DataFrame(
            columns=['identifier', 'modality']+self.bbox_cols
        )
        disk_idx = 0

        with h5py.File(self.output_file, 'w') as ff:
            for pid in self.case_df.index:
                for mod in self.modalities:
                    print(f'{pid}-{mod}')
                    try:
                        data, bbox = self.get_case(pid, mod, rtn_bbox=True)
                        width = data.shape[1]

                        # left body side is on the right of the image for LPS!!
                        # ris: right image side
                        # lis: left image side
                        ris = self.pathology_of_side(pid, 'L')
                        ff.create_dataset(
                            f'{pid}/{ris}/{mod}',
                            data=data[:, width//2:, :]
                        )
                        ff[f'{pid}'].attrs['ris'] = ris
                        lis = self.pathology_of_side(pid, 'R')
                        ff.create_dataset(
                            f'{pid}/{lis}/{mod}',
                            data=data[:, :width//2, :],
                        )
                        ff[f'{pid}'].attrs['lis'] = lis
                    except (Exception, ArithmeticError) as e: 
                        print(f'failed: {e}')
                        continue

                    disk_df.loc[disk_idx] = [pid, mod, *bbox]
                    disk_idx += 1

        disk_df.to_csv(
            self.output_file.parent / (self.output_file.stem + '.csv'),
            index=False
        )

if __name__ == '__main__':
    
    spacing = (0.75, 0.75, 1.0)
    min_size_mm = [False]*3 

    rescale = False
    crop = 'before'
    modalities=['t1_non_FS', 't1_FS', 'segmentation']
    normalize=['t1_non_FS', 't1_FS']
    bias_correction = []
    std = 0.25
    mean = 0.5

    mean_str = str(mean).replace('.', '')
    std_str = str(std).replace('.', '')

    space_str = 'x'.join([str(ii) for ii in spacing]).replace('.', '')
    size_str = '200x200x50' #'x'.join([str(int(ii)) for ii in min_size_mm]).replace('.', '')

    bbox_file = f'./../../data/bboxes/breast_tissue_from_sgmt.csv'
    path_file = './../../data/file_paths.csv'
    side_file = './../../data/TCIA/Duke-Breast-Cancer-MRI/Clinical_and_Other_Features.xlsx'

    output_file = f'./../../data/patch_data/complete_clc_mean{mean_str}_'\
        f'std{std_str}_space{space_str}_size{size_str}/patches.h5'

    patch_gen = GetPatches(
        bbox_file=bbox_file,
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

    def normalize_image(data, std, mean):
        f = sitk.NormalizeImageFilter()
        data = f.Execute(data)
        return (std*data)+mean
    
    patch_gen.normalize_image = lambda img: normalize_image(img, std=std, mean=mean)

    # testing
    #patch_gen.case_df = patch_gen.case_df.iloc[:2]
    
    patch_gen.to_disk()

    output_file = Path(output_file)
    script = output_file.parent / (output_file.stem + '.py')
    with open(script, 'w') as ff: 
        ff.write(Path(__file__).read_text())

    to_dump = {
        'spacing': spacing,
        'min_size_mm': min_size_mm,
        'rescale': rescale,
        'normalize': normalize,
        'std_value': std,
        'mean_value': mean,
        'bbox_file': bbox_file,
    }

    dump_file = output_file.parent / (output_file.stem + '.pkl')
    with open(dump_file, 'wb') as ff:
        pickle.dump(to_dump, ff)

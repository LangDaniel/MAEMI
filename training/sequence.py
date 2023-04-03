from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):

    def __init__(self, args, with_label=True):
        self.args = args

        patch_file = Path(args.patch_file)
        disk_df = pd.read_csv(patch_file.parent / (patch_file.stem + '.csv'))
        disk_ids = disk_df['identifier'].values

        label_df = pd.read_csv(args.label_file)
        label_df = label_df[label_df['identifier'].isin(disk_ids)]

        # for validation augmentation add column with augmentation name
        #if not args.training and args.augment:
        #    aug_df = pd.DataFrame()
        #    for aug in (['unity'] + args.augment):
        #        tmp_df = label_df.copy()
        #        tmp_df['augmentation'] = aug
        #        aug_df = aug_df.append(tmp_df)
        #    label_df = aug_df

        self.label_df = label_df.reset_index(drop=True)

        if not self.args.training:
            self.label_df = self.label_df.sort_values(by=['identifier'])

        self.total_count = len(self.label_df)
        if not self.total_count:
            raise ValueError('no cases were found: check label naming')
        print(f'len {self.total_count}')

    # augmentation 
    # ---------------------------------------------------------------- 
    @staticmethod
    def rescale_linear(data, scaling):
        [in_low, in_high], [out_low, out_high] = scaling

        m = (out_high - out_low) / (in_high - in_low)
        b = out_low - m * in_low
        data = m * data + b
        data = np.clip(data, out_low, out_high)
        return data
        
    @staticmethod
    def add_gaussian_noise(data, noise_var):
        variance = np.random.uniform(noise_var[0], noise_var[1])
        data = data + np.random.normal(0.0, variance, size=data.shape) 
    
        return data

    @staticmethod
    def gaussian_blur(data, sigma):
        sig = np.random.uniform(sigma[0], sigma[1])
        return gaussian_filter(data, sig)

    @staticmethod
    def add_offset(data, var):
        off = np.random.uniform(var[0], var[1])
        data = data + off

        return data

    @staticmethod
    def resample(data, xy_zoom, z_zoom):
        fac = [xy_zoom, xy_zoom, z_zoom]

        return zoom(data, zoom=fac, order=1)

    def get_aug_func(self):
        ''' returns a list with all the augmentations '''
        aug = []
        if self.args.augment.flip:
            # flip on sagittal plane
            if np.random.rand() < 0.5:
                aug.append(
                    lambda data: np.flip(data, axis=0)
                )
            # flip on coronal plane
            if np.random.rand() < 0.5:
                aug.append(
                    lambda data: np.flip(data, axis=1)
                )

        if self.args.augment.rot:
            # rotate k*90 degree
            rot_k = np.random.randint(0, 4)
            if rot_k:
                aug.append(
                    lambda data: np.rot90(
                        data, axes=(0, 1), k=rot_k
                    )
                )

        if self.args.augment.zoom:
            xy_zoom = np.random.uniform(
                self.args.augment.zoom[0],
                self.args.augment.zoom[1],
            )
            z_zoom = np.random.uniform(
                1.,
                self.args.augment.zoom[1],
            )
            aug.append(
                lambda data: self.resample(
                    data,
                    xy_zoom,
                    z_zoom,
                )
            )

        if self.args.augment.offset:
            aug.append(
                lambda data: self.add_offset(
                    data,
                    self.args.augment.offset
                )
            ) 

        if self.args.augment.blur:
            if self.args.augment.blur.type.lower() == 'gaussian':
                aug.append(
                    lambda data: self.gaussian_blur(
                        data,
                        self.args.augment.blur.sigma
                    )
                )
            else:
                raise ValueError('blurring not registered')

        if self.args.augment.noise:
            if self.args.augment.noise.type.lower() == 'gaussian':
                aug.append(
                    lambda data: self.add_gaussian_noise(
                        data,
                        self.args.augment.noise.variance
                    )
                )
            else:
                raise ValueError('noise not registered')

        return aug

    @staticmethod
    def get_valid_auc_func(augmentation):
        ''' returns a list with all the augmentations '''
        if augmentation == 'unity':
            return lambda data: data
        if augmentation == 'flip_coronal':
            return lambda data: np.flip(data, axis=0)
        if augmentation == 'flip_sagittal':
            return lambda data: np.flip(data, axis=1)
        if augmentation == 'rot_90':
            return lambda data: np.rot90(data, axes=(0, 1), k=1)
        if augmentation == 'zoom_1.1':
            return lambda data: zoom(data, zoom=1.1, order=1)
        if augmentation == 'zoom_0.9':
            return lambda data: zoom(data, zoom=0.9, order=1)
        raise ValueError('validation augmentation not listed')


    def augmentation(self, data, aug_func):
        for func in aug_func:
            data = func(data)
        return data
    # ---------------------------------------------------------------- 

    # crop patches
    # ---------------------------------------------------------------- 
    def get_index(self, patch_shape):
        init = np.zeros(len(patch_shape))
        fin = np.zeros(len(patch_shape))

        # use random indices for training
        if self.args.training:
            for ii in range(0, len(patch_shape)):
                ## choose a range to pick from
                # mh: high margin, ml: left margin
                ml, mh = self.args.margin[ii]
                m = ml + mh
                if m:
                    m_max = patch_shape[ii] - self.args.shape[ii] - 1
                    m_max = max(0, m_max)
                    ml_max = ml * (m_max / m)
                    mh_max = m_max - ml
                    mh_max = max(0, mh_max)
                    ml = min(ml_max, ml)
                    mh = min(mh_max, mh)
                low = ml
                high = patch_shape[ii] - self.args.shape[ii] - mh + 1
                #diff = patch_shape[ii] - self.args.shape[ii] - 2*m
                #m += min(diff//2, 0)
                #low = m
                #high = patch_shape[ii] - self.args.shape[ii] - m + 1
                init[ii] = np.random.randint(
                    low=low,
                    high=high,
                    #low=0,
                    #high=patch_shape[ii] - self.args.shape[ii] + 1
                )

        # center for validation
        else:
            for ii in range(0, len(patch_shape)):
                init[ii] = (patch_shape[ii] - self.args.shape[ii]) // 2 

        fin = init + np.array(self.args.shape)

        init = init.astype(int)
        fin = fin.astype(int)

        out = np.empty(len(init) + len(fin))
        out[::2] = init
        out[1::2] = fin

        return out.astype(int)

        
    def crop_patch(self, data, idx):
        if len(idx) == 6:
            return data[idx[0]:idx[1], idx[2]:idx[3], idx[4]:idx[5]]
        elif len(idx) == 4:
            return data[idx[0]:idx[1], idx[2]:idx[3]]
        raise ValueError('patch_shape not implemented')
    # ---------------------------------------------------------------- 

    # get data
    # ---------------------------------------------------------------- 

    @staticmethod
    def to_slice(shape):
        slz = []
        for shp in shape:
            slz.append(np.s_[:shp])
        return tuple(slz)
    
    def zero_pad(self, img, shape):
        # zero pad if too small
        padw = np.zeros(6)
        for ax in range(img.ndim):
            diff = shape[ax] - img.shape[ax]
            if diff < 0:
                continue
            padw[::2][ax] = int(np.ceil(diff/2)) 
            padw[1::2][ax] = diff//2
        padw = padw.reshape(3, 2).astype(int)
        if padw.any():
            img = np.pad(img, pad_width=padw)
        return img

    def assert_size(self, img):
        img = img[self.to_slice(self.args.shape)]
        return img

    def get_modality(self, pid, modality):
        with h5py.File(self.args.patch_file, 'r') as ff:
            if self.args.pathology:
                data = ff[f'{pid}'][self.args.pathology][modality][:]
            else:
                data = ff[f'{pid}'][modality][:]

        # rescale intensity values linear
        if self.args.scale:
            data = self.rescale_linear(
                data,
                self.args.scale,
            )

        return data

        
    def get_patch(self, idx):
        pid = self.label_df.loc[idx]['identifier']

        patch = np.empty((len(self.args.modalites), *self.args.shape)) 

        for ii, mod in enumerate(self.args.modalites):
            modality = self.get_modality(pid, mod) 
            modality = self.zero_pad(modality, self.args.shape)
            if not ii:
                shape = modality.shape
                index = self.get_index(shape) 
                aug_func = self.get_aug_func()
            else:
                modality = self.zero_pad(modality, shape)
            p = self.crop_patch(modality, index)
            p = self.assert_size(p)
            if self.args.training:
                p = self.augmentation(p, aug_func)
            patch[ii] = p 

        patch = np.nan_to_num(patch)
        if self.args.max_value or self.args.min_value:
            patch = np.clip(
                patch,
                a_min=self.args.min_value,
                a_max=self.args.max_value,
            )

        return patch.astype(np.float32)

    def get_label(self, idx):
        label = self.label_df.loc[idx][self.args.label_column]
        map_dict = self.args.label_mapping.__dict__
        if map_dict:
            label = map_dict[label]
        if self.args.one_hot:
            n_classes = len(np.unique(list(map_dict.values())))
            label = np.eye(n_classes)[int(label)].astype(np.float32)
        else:
            label = np.array(label, dtype=np.int)

        return label
    # ---------------------------------------------------------------- 
    
    # torch Dataset stuff 
    # ---------------------------------------------------------------- 
    def __len__(self):
        ''' cases involved '''
        return self.total_count

    def get_case(self, idx, information=False):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        patch = self.get_patch(idx)
        if information:
            info = self.label_df.loc[idx][information].to_dict()
            return patch, info
        return patch

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        patch = self.get_patch(idx)
        # artifically add color channel
        #patch = np.repeat(patch, 4, 0)
        label = self.get_label(idx)
        
        return patch, label
    # ---------------------------------------------------------------- 

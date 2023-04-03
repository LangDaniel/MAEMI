import numpy as np

class SliceIterator(object):
    
    def __init__(self, vol_shape, ptc_shape, stride, offset=[0, 0, 0]): 
        steps = self.get_steps(
            vol_shape,
            ptc_shape,
            stride,
        )
        
        self.slices = self.get_slices(steps, ptc_shape, offset)
        self.count = 0

    def reset(self):
        self.count = 0
    
    @staticmethod
    def get_steps_per_ax(ax_size, patch_size, stride):
        steps = np.arange(patch_size, ax_size, stride)
        return np.append(steps, [ax_size])
    
    def get_steps(self, vol_shape, ptc_shape, stride):
        steps = []
        for ii in range(len(vol_shape)):
            steps.append(
                self.get_steps_per_ax(vol_shape[ii], ptc_shape[ii], stride[ii])
            )
        return steps
    
    @staticmethod
    def get_slices(steps, patch_size, offset):
        limit = np.prod([len(ii) for ii in steps])
        slices = []
        
        idx = 0
        while idx < limit:
            xi, yi, zi = 0, 0, 0
            for xf in steps[0]:
                for yf in steps[1]:
                    for zf in steps[2]:
                        xi = xf - patch_size[0] + offset[0]
                        xi = max(xi, 0)
                        xff = xf + offset[0]
                        yi = yf - patch_size[1] + offset[1]
                        yi = max(yi, 0)
                        yff = yf + offset[1]
                        zi = zf - patch_size[2] + offset[2]
                        zi = max(zi, 0)
                        zff = zf + offset[2]
                        slices.append(
                            np.s_[xi:xff, yi:yff, zi:zff]
                        )
                        idx += 1
        return slices
    
    def __iter__(self):
        return self
    
    def next(self):
        if self.count < (len(self.slices)):
            self.count += 1
            return self.slices[self.count - 1]
        raise StopIteration()
    
    def __next__(self):
        return self.next()

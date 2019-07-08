import torch
import torch.utils.data as data
import os
import nibabel as nii
import numpy as np
import random
from scipy.ndimage import zoom
import math
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from datetime import datetime
from scipy.ndimage.filters import median_filter
from scipy.ndimage import affine_transform

import loader_helper

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

import model

#https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
def elastic_transform(image, alpha, sigma, order=3, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert len(image.shape) == 3

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * (alpha/2.5)

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')

    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(z + dz, (-1, 1))

    #print(dx[:100,0])

    return map_coordinates(image, indices, order=order, mode='reflect').reshape(shape)

def normalize(image, mask):
    """
    perform data normalization
    :param image: input nd array
    :param mask: corresponding foreground mask
    :return: normalized array
    """
    ret = image.copy()
    image_masked = np.ma.masked_array(ret, ~(mask))
    ret[mask] = ret[mask] - np.mean(image_masked)
    ret[mask] = ret[mask] / np.var(image_masked) ** 0.5

    ret[~mask] = -100.

    return ret


class SimpleReader(data.Dataset):
    def __init__(self, path, patch_size, series = None, multiplier=1, patches_from_single_image=1):
        super(SimpleReader, self).__init__()
        self.path = path
        self.patch_size = patch_size
        self.multiplier = multiplier
        self.patches_from_single_image = patches_from_single_image

        if series is None:
            self.series = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        else:
            self.series = series

        self.series.sort()

        self.labels_location = []
        self.__cache()

        self.real_length = len(self.series)

        self.patches_from_current_image = self.patches_from_single_image
        self.current_image_index = 0

        self.__load(self.current_image_index)

        print(len(self.series))


    def __cache(self):
        # cache locations of the labels (bounding boxes) inside the images
        for f in self.series:

            image, label, affine = loader_helper.read_multimodal(self.path, f, True)


            bbox = loader_helper.bbox3(label>0)


            borders = np.array(label.shape)
            borders_low = np.array(self.patch_size) / 2.0 + 1
            borders_high = borders - np.array(self.patch_size) / 2.0 - 1

            bbox[0] = borders_low#np.maximum(bbox[0]-100, borders_low)
            bbox[1] = borders_high#np.minimum(bbox[1]+100, borders_high)

            self.labels_location.append(bbox)


    def __load(self, index):
        if self.patches_from_current_image > self.patches_from_single_image:
            self.patches_from_current_image = 0
            self.current_image_index = index

            self.image, self.label, affine = loader_helper.read_multimodal(self.path, self.series[index], True)


            mask = self.image > 0
            num_voxels = np.sum(mask,axis=(1,2,3))

            #print(self.image[mask].shape)

            mean = np.sum(self.image,axis=(1,2,3)) / num_voxels
            mean2 = np.sum(self.image ** 2,axis=(1,2,3)) / num_voxels

            std = np.sqrt(mean2 - mean * mean)

            #std1 = self.image.std(axis=(1,2,3))

            self.image = self.image / std.reshape((self.image.shape[0],1,1,1))#self.image.std(axis=(1,2,3), keepdims=True)#(self.image - self.image.mean(axis=(1,2,3), keepdims=True)) / self.image.std(axis=(1,2,3), keepdims=True)

        self.patches_from_current_image += 1

    def __getitem__(self, index):
        index = index % self.real_length
        self.__load(index)
        center = np.random.rand(3)

        bbox = self.labels_location[self.current_image_index]

        center = center * (bbox[1] - bbox[0]) + bbox[0]
        left_bottom = center - np.array(self.patch_size) / 2.0
        left_bottom = left_bottom.astype(np.int32)

        data_out = self.image[:,left_bottom[0]:left_bottom[0] + self.patch_size[0],
                   left_bottom[1]:left_bottom[1] + self.patch_size[1],
                   left_bottom[2]:left_bottom[2] + self.patch_size[2]]

        label_out = self.label[left_bottom[0]:left_bottom[0] + self.patch_size[0],
                    left_bottom[1]:left_bottom[1] + self.patch_size[1],
                    left_bottom[2]:left_bottom[2] + self.patch_size[2]]


        seed = datetime.now().microsecond
        sigma = random.random()*20 + 10
        alpha = random.random()*4000 + 200

        x_scale = 0.7 + random.random()*0.6
        y_scale = 0.7 + random.random()*0.6

        data_out = affine_transform(data_out,(1,x_scale,y_scale,1),order=1)
        #data_out = np.stack([elastic_transform(data_out[i], alpha, sigma, 1, np.random.RandomState(seed)) for i in range(data_out.shape[0])],axis=0)

        label_out = affine_transform(label_out, (x_scale, y_scale, 1), order=0)
        #label_out = elastic_transform(label_out, alpha, sigma, 0, np.random.RandomState(seed))

        #label_out = np.eye(2)[(label_out>0).astype(np.int32) ].transpose((3,0,1,2))
        label_out = (label_out > 0)[None]

        if random.random() > 0.5:
            data_out = data_out[:,::-1,:,:].copy()
            label_out = label_out[:,::-1,:,:].copy()


        if random.random() > 0.5:
            data_out = data_out[:,:,::-1,:].copy()
            label_out = label_out[:,:,::-1,:].copy()

        #if random.random() > 0.5:
        #    data_out = data_out.transpose((0,1,3,2)).copy()
        #    label_out = label_out.transpose((0,1,3,2)).copy()

        #if random.random() > 0.5:
        #    data_out = data_out[:,:,:,::-1].copy()
        #    label_out = label_out[:,:,:,::-1].copy()


        #data_out = data_out / 500

        data_out = data_out * (0.6+random.random()*0.8)

        data_out = data_out + 1.2*(random.random() - 0.5)

        labels_torch = torch.from_numpy(label_out.copy()).float()


        return [torch.from_numpy(data_out).float(),
                ], \
                [
                labels_torch,]


    def __len__(self):
        return int(self.multiplier*self.real_length)


class FullReader(data.Dataset):
    def __init__(self, path, series = None):
        super(FullReader, self).__init__()
        self.path = path

        if series is None:
            self.series = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        else:
            self.series = series

        self.series.sort()

    @staticmethod
    def get_data_filename(path, series):
        return os.path.join(path, series,series+'.nii.gz')
        #return os.path.join('/home/localadmin/segthor/data/train_resampled2_2_25_normalized', series + '.npy')

    @staticmethod
    def get_label_filename(path, series):
        return os.path.join(path, series,'GT.nii.gz')

    def __getitem__(self, index):

        image, label, affine = loader_helper.read_multimodal(self.path, self.series[index], True)

        #image = image - image.mean()
        #image = image / image.var() ** 0.5

        #image = image / 1000.

        old_shape = image.shape
        new_shape = tuple([loader_helper.closest_to_k(i,16) for i in old_shape[1:]])
        new_image = np.full(shape=(old_shape[0],)+new_shape, fill_value=0., dtype=np.float32)
        new_label = np.zeros(shape=new_shape, dtype=np.float32)

        new_image[:,:old_shape[1],:old_shape[2],:old_shape[3]] = image
        new_label[:old_shape[1],:old_shape[2],:old_shape[3]] = label

        mask = new_image > 0
        num_voxels = np.sum(mask, axis=(1, 2, 3))

        mean = np.sum(new_image, axis=(1, 2, 3)) / num_voxels
        mean2 = np.sum(new_image ** 2, axis=(1, 2, 3)) / num_voxels

        std = np.sqrt(mean2 - mean * mean)

        new_image = new_image / std.reshape((new_image.shape[0], 1, 1, 1))
        #new_image = new_image / new_image.std(axis=(1, 2, 3),keepdims=True) #(new_image - new_image.mean(axis=(1, 2, 3), keepdims=True)) / new_image.std(axis=(1, 2, 3),keepdims=True)

        #new_label_out = (np.eye(2)[(new_label>0).astype(np.int32)]).transpose((3,0,1,2))

        new_label_out = (new_label > 0)[None]

        labels_torch = torch.from_numpy(new_label_out.copy()).float()

        return [torch.from_numpy(new_image).float(),
                ], \
                [  labels_torch,
                   ]

    def __len__(self):
        return len(self.series)

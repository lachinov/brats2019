import torch
import torch.utils.data as data
import os
import nibabel as nii
import numpy as np
import random
from scipy.ndimage import zoom
import augment
import math
from scipy.ndimage import gaussian_filter
from skimage.measure import block_reduce

import loader_helper

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

        print(self.series)

    @staticmethod
    def get_data_filename(path, series):
        return os.path.join(path, series,series+'.nii.gz')
        #return os.path.join('/home/localadmin/segthor/data/train_resampled2_2_25_normalized', series + '.npy')

    @staticmethod
    def get_label_filename(path, series):
        return os.path.join(path, series,'GT.nii.gz')

    def __cache(self):
        # cache locations of the labels (bounding boxes) inside the images
        for f in self.series:
            label = loader_helper.read_nii(self.get_label_filename(self.path,f))
            image = loader_helper.read_nii(self.get_data_filename(self.path,f))

            #bboxes = []
            #for i in range(5):
            #    bboxes.append(loader_helper.bbox3(label == i))

            #bbox = bboxes[0]
            #bbox[0] = bbox[0] - 64
            #bbox[1] = bbox[1] + 64

            #for b in bboxes:
            #    b[0] = b[0] - np.array(self.patch_size) / 2.0
            #    b[1] = b[1] + np.array(self.patch_size) / 2.0

            #    bbox[0] = np.maximum(bbox[0], b[0])
            #    bbox[1] = np.minimum(bbox[1], b[1])


            #bbox = loader_helper.bbox3(label==1)
            bbox = loader_helper.bbox3(image > -500)


            borders = np.array(label.shape)
            borders_low = np.array(self.patch_size) / 2.0
            borders_high = borders - np.array(self.patch_size) / 2.0

            bbox[0] = np.maximum(bbox[0], borders_low)
            bbox[1] = np.minimum(bbox[1], borders_high)

            self.labels_location.append(bbox)


    def __load(self, index):
        if self.patches_from_current_image > self.patches_from_single_image:
            self.patches_from_current_image = 0
            self.current_image_index = index
            filename = self.get_data_filename(self.path, self.series[index])
            labelname = self.get_label_filename(self.path, self.series[index])
            self.image = loader_helper.read_nii(filename)
            self.label = np.eye(5)[loader_helper.read_nii(labelname).astype(np.int32)].transpose((3,0,1,2))

            #self.transformation = augment.create_identity_transformation(self.patch_size)
            #self.transformation += augment.create_elastic_transformation(
            #    self.patch_size,
            #    control_point_spacing=20,
            #    jitter_sigma=random.random() * 5.)

            #self.transformation += augment.create_rotation_transformation(
            #    self.patch_size,
            #    math.pi / 6 * random.random())

            self.image = self.image - self.image.mean()
            self.image = self.image / self.image.var() ** 0.5
            
            #distance_maps = []
            
            #for i in range(4):
            #    l = self.label[i+1]
            #    distance_maps.append(gaussian_filter(l*1000.0, sigma=10)[None])
            #self.distance = np.concatenate(distance_maps, axis=0)

            #self.label[self.label > 1] = 0

            #self.image = self.image * (self.image > 300.)

            #self.image = normalize(self.image, self.image > -500)
            #self.image = self.image / 1000.

        self.patches_from_current_image += 1

    def __getitem__(self, index):
        index = index % self.real_length
        self.__load(index)
        center = np.random.rand(3)

        bbox = self.labels_location[self.current_image_index]

        center = center * (bbox[1] - bbox[0]) + bbox[0]
        left_bottom = center - np.array(self.patch_size) / 2.0
        left_bottom = left_bottom.astype(np.int32)

        #print(self.image.shape, self.label.shape, self.distance.shape)
        
        data_out = self.image[None, left_bottom[0]:left_bottom[0] + self.patch_size[0],
                           left_bottom[1]:left_bottom[1] + self.patch_size[1],
                           left_bottom[2]:left_bottom[2] + self.patch_size[2]].transpose(0,3,2,1)

        #if np.any(np.array(data_out.shape) != np.array([1,192,192,128])):
        #    print(self.series[self.current_image_index], bbox, self.image.shape, center, left_bottom, self.patch_size, data_out.shape)

        label_out = self.label[:,left_bottom[0]:left_bottom[0] + self.patch_size[0],
                            left_bottom[1]:left_bottom[1] + self.patch_size[1],
                            left_bottom[2]:left_bottom[2] + self.patch_size[2]].transpose((0,3,2,1))

        #distance_out = self.distance[:,left_bottom[0]:left_bottom[0] + self.patch_size[0],
        #                    left_bottom[1]:left_bottom[1] + self.patch_size[1],
        #                    left_bottom[2]:left_bottom[2] + self.patch_size[2]].transpose((0,3,2,1))
        
        if random.random() > 0.5:
            data_out = data_out[:,:,:,::-1].copy()
            label_out = label_out[:,:,:,::-1].copy()

        if random.random() > 0.5:
            data_out = data_out[:,:,::-1,:].copy()
            label_out = label_out[:,:,::-1,:].copy()

        #if random.random() > 0.5:
        #    data_out = data_out.transpose((0,1,3,2)).copy()
        #    label_out = label_out.transpose((0,1,3,2)).copy()

        #data_out[0] = augment.apply_transformation(data_out[0], self.transformation)

        #for i in range(1, label_out.shape[0]):
        #    label_out[i] = augment.apply_transformation(label_out[i], self.transformation) > 0.5

        #if random.random() > 0.5:
        #    data_out = data_out[:,:,:,::-1].copy()
        #    label_out = label_out[:,:,:,::-1].copy()

        data_out = data_out + (random.random() - 0.5)

        data_out = data_out * (0.9+random.random()*0.2)

        #attention_fg = label_out[1:].max(axis=(0, 2, 3)).reshape((1, -1, 1, 1)).copy()
        #attention_es = label_out[1].max(axis=(1, 2)).reshape((1, -1, 1, 1)).copy()
        #esophagus = np.max(label_out[0]==0,axis=(0,1)).reshape((1, 1,1,-1)).astype(np.float32)
        #esophagus_ds = zoom(esophagus, (1,1,1,1.0/4), order=0)
        #print(esophagus)

        #label_eso = label_out[1]
        #label_eso_reg = gaussian_filter(label_eso*100.0, sigma=10)
        #label_ds = block_reduce(label_eso,(4,4,4),np.max)
        #label_ds = zoom(label_eso, (1.0/8,1.0/8,1.0/8), order=0)
        #label_bg = 1.0 - label_eso

        #print(np.any(label_eso+label_bg>1.))
        
        #print(label_out.shape)
        return [torch.from_numpy(data_out).float(),
                #torch.from_numpy(attention_es).float(),
                ], \
                [
                torch.from_numpy(label_out).float(),
                #torch.from_numpy(label_ds[None]).float(),
                #torch.from_numpy(distance_out).float(),
                #torch.from_numpy(attention_es).float(),
                #torch.from_numpy(esophagus).float(),
                #torch.from_numpy(esophagus_ds).float()
                ]


    def __len__(self):
        return self.multiplier*self.real_length


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

        filename = self.get_data_filename(self.path, self.series[index])
        labelname = self.get_label_filename(self.path, self.series[index])

        image = loader_helper.read_nii(filename)
        label = loader_helper.read_nii(labelname)

        #image = image - image.mean()
        #image = image / image.var() ** 0.5

        #image = image / 1000.

        old_shape = image.shape
        new_shape = tuple([loader_helper.closest_to_k(i,16) for i in old_shape])
        new_image = np.full(shape=new_shape, fill_value=-1000., dtype=np.float32)
        new_label = np.zeros(shape=new_shape, dtype=np.float32)

        new_image[:old_shape[0],:old_shape[1],:old_shape[2]] = image
        new_label[:old_shape[0],:old_shape[1],:old_shape[2]] = label

        #new_label[new_label > 1 ] = 0

        #print(new_image.shape)
        #new_image = new_image * (new_image > 300.)
        #new_image = normalize(new_image, new_image > -500)

        new_image = new_image - new_image.mean()
        new_image = new_image / new_image.var() ** 0.5


        new_image = new_image.transpose((2,1,0))
        new_label_out = (np.eye(5)[new_label.astype(np.int32)]).transpose((3, 2, 1, 0))

        #attention_fg = new_label_out[1:].max(axis=(0,2,3)).reshape((1,-1,1,1)).copy()
        #attention_es = new_label_out[1].max(axis=(1,2)).reshape((1,-1,1,1)).copy()

        #label_eso = new_label_out[1,None]
        #label_bg = 1.0 - label_eso


        distance_maps = []
            
        for i in range(4):
            l = new_label_out[i+1]
            distance_maps.append(gaussian_filter(l*1000.0, sigma=10)[None])
        distance = np.concatenate(distance_maps, axis=0)
        
        #label_eso = new_label_out[1]
        #label_eso_reg = gaussian_filter(label_eso*100.0, sigma=10)
        #label_ds = block_reduce(label_eso,(4,4,4),np.max)
        #label_ds = zoom(label_eso, (1.0/8,1.0/8,1.0/8), order=0)


        return [torch.from_numpy(new_image[None,:,:,:]).float(),
                #torch.from_numpy(attention_es).float(),
                ], \
                [  torch.from_numpy(new_label_out).float(),
                   #torch.from_numpy(label_ds[None]).float(),
                   torch.from_numpy(distance).float(),
                   #torch.from_numpy(attention_es).float(),
                   ]

    def __len__(self):
        return len(self.series)
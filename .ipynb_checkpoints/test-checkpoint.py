import loader_helper
import dataloader
import os
import numpy as np
import torch
import train
import argparse
import nibabel as nii
from scipy import ndimage
from scipy.ndimage import zoom
from skimage.filters import threshold_otsu
import pickle
from scipy.ndimage.filters import median_filter


parser = argparse.ArgumentParser(description="PyTorch SegTHOR")
#parser.add_argument("--test_path", default="", type=str, help="path to train data")
parser.add_argument("--name", default="test", type=str, help="experiment name")
parser.add_argument("--models_path", default="/models", type=str, help="path to models folder")


new_scale = np.array([2.0,2.0,2.5])

def m(coords, new_scale, old_scale, translate):
    return coords* new_scale / old_scale + translate

def warp(data, new_scale, old_scale, translation, output_shape, cval, order):
    coordinates = np.zeros(shape=(3,)+output_shape,dtype=float)
    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            for k in range(output_shape[2]):
                coordinates[:,i,j,k]=m(np.array([i,j,k]), new_scale, old_scale, translation)
    return ndimage.map_coordinates(data, coordinates, order=order, cval=cval, mode='constant')


if __name__ == '__main__':
    opt = parser.parse_args()
    print(torch.__version__)
    print(opt)
    path = '../../data/test_experimental'
    output_path = '../../data/output_051'

    trainer = train.Trainer(name=opt.name, models_root=opt.models_path, rewrite=False, connect_tb=False)
    trainer.load_best()
    trainer.model = trainer.model.module.cpu()
    # trainer.model.eval()
    trainer.state.cuda = False

    files = os.listdir(path)
    files = [f for f in files if (f.startswith('Patient') and os.path.isfile(os.path.join(path,f)))]
    files.sort()

    print(files)

    for f in files:
        image_name = f
        data_header = loader_helper.read_nii_header(os.path.join(path,f))
        affine = data_header.affine

        desc = pickle.load(open(os.path.join(path,'desc'+f+'.p'),'rb'))

        data_crop = np.array(data_header.get_data())
        print(data_crop.shape)
        #data_crop = median_filter(data_crop, 3)
#=========================================
        old_shape_crop = data_crop.shape
        new_shape_crop = tuple([loader_helper.closest_to_k(i, 16) for i in old_shape_crop])
        diff = np.array(new_shape_crop) - np.array(old_shape_crop)
        pad_left = diff // 2
        pad_right = diff - pad_left

        #new_data_crop = np.full(shape=new_shape_crop, fill_value=-1000., dtype=np.float32)
        #new_data_crop[:old_shape_crop[0], :old_shape_crop[1], :old_shape_crop[2]] = data_crop

        new_data_crop = np.pad(data_crop, pad_width=tuple([(pad_left[i], pad_right[i]) for i in range(3)]),
                           mode='reflect')

        #mask = new_data_crop > -100
        #num_voxels = np.sum(mask)

        #mean = np.sum(new_data_crop[mask]) / num_voxels
        #mean2 = np.sum(new_data_crop[mask] ** 2) / num_voxels

        #std = np.sqrt(mean2 - mean * mean)

        #new_data_crop = (new_data_crop - mean) / (std + 1e-6)

        mean = -303.0502877950004
        mean2 = 289439.0029958802
        std = np.sqrt(mean2 - mean * mean)

        new_data_crop = (new_data_crop - mean) / std

        #new_data_crop = new_data_crop.transpose((2,1,0))

        new_data_crops = []

        new_data_crops.append(new_data_crop)
        new_data_crops.append(new_data_crop[::-1,:,:].copy())
        new_data_crops.append(new_data_crop[:,::-1,:].copy())
        new_data_crops.append(new_data_crop[::-1,::-1,:].copy())
        ##new_data_crops.append(new_data_crop+0.1)
        ##new_data_crops.append(new_data_crop[:,::-1,:].copy()+0.1)
        ##new_data_crops.append(new_data_crop[:,:,::-1].copy()+0.1)
        ##new_data_crops.append(new_data_crop[:,::-1,::-1].copy()+0.1)

        # tta
        outputs = []
        for new_data_crop in new_data_crops:
            new_data_crop = torch.from_numpy(new_data_crop[None, None, :, :, :]).float()

            output = trainer.predict([[new_data_crop], ])

            output = output[0].cpu().detach().numpy()
            output_crop = output[0]

            outputs.append(output_crop)

        outputs[1] = outputs[1][:, ::-1, :, :].copy()
        ##outputs[5] = outputs[5][:, :, ::-1, :].copy()

        outputs[2] = outputs[2][:, :, ::-1, :].copy()
        ##outputs[6] = outputs[6][:, :, :, ::-1].copy()

        outputs[3] = outputs[3][:, ::-1, ::-1, :].copy()
        ##outputs[7] = outputs[7][:, :, :, ::-1].copy()

        #for idx, o in enumerate(outputs):
        #    outputs[idx] = o.transpose((0, 3, 2, 1))

        #for idx, o in enumerate(outputs):
        #    o = np.argmax(o, axis=0).astype(np.int32)
        #    output_header = nii.Nifti1Image(o, affine)
        #    nii.save(output_header, os.path.join(output_path, image_name[:-7] + str(idx) + '.nii'))

        # break

        output_crop = sum(outputs) / len(outputs)
        #output_crop = output_crop[:, :old_shape_crop[0], :old_shape_crop[1], :old_shape_crop[2]]

        output_crop = output_crop[:, pad_left[0]:-pad_right[0] or None, pad_left[1]:-pad_right[1] or None,
                 pad_left[2]:-pad_right[2] or None]

        # ==========================
        bbox = desc['bbox']

        new_label = np.zeros(shape=(4,)+desc['new_shape'])
        #new_label[0] = 1.0
        new_label[:,bbox[0, 0]:bbox[1, 0], bbox[0, 1]:bbox[1, 1], bbox[0, 2]:bbox[1, 2]] = output_crop


        #output = (np.argmax(output, axis=1) + 1)*(np.max(output,axis=1) > 0.5).astype(np.int32)
        #output = output[0,:old_shape[0], :old_shape[1], :old_shape[2]]

# ==========================

        scale_factor = np.array(desc['old_shape']) / np.array(new_label[0].shape)
        old_labels = [zoom(new_label[i], scale_factor, order=3, mode='constant', cval=0)[None] for i in range(4)]

        old_label = np.concatenate(tuple(old_labels),axis=0)
        old_label = ((np.argmax(old_label, axis=0) + 1) * (np.max(old_label, axis=0) > 0.7)).astype(np.int32)

        output_header = nii.Nifti1Image(old_label, affine)

        nii.save(output_header, os.path.join(output_path,image_name[:-7]+'.nii'))


        print(f, np.all(np.array(old_label.shape) == np.array(desc['old_shape'])), old_label.shape, desc['old_shape'])

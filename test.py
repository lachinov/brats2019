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

series_val = ['BraTS19_2013_0_1',
              'BraTS19_2013_12_1',
              'BraTS19_2013_16_1',
              'BraTS19_2013_2_1',
              'BraTS19_2013_23_1',
              'BraTS19_2013_26_1',
              'BraTS19_2013_29_1',
              'BraTS19_CBICA_AAB_1',
              'BraTS19_CBICA_AAP_1',
              'BraTS19_CBICA_AMH_1',
              'BraTS19_CBICA_AQD_1',
              'BraTS19_CBICA_ATX_1',
              'BraTS19_CBICA_AZH_1',
              'BraTS19_CBICA_BHB_1',
              'BraTS19_TCIA12_101_1',
              'BraTS19_TCIA01_150_1',
              'BraTS19_TCIA10_152_1',
              'BraTS19_TCIA04_192_1',
              'BraTS19_TCIA08_205_1',
              'BraTS19_TCIA06_211_1',
              'BraTS19_TCIA02_222_1',
              'BraTS19_TCIA12_298_1',
              'BraTS19_TCIA13_623_1',
              'BraTS19_CBICA_ANV_1',
              'BraTS19_CBICA_BBG_1',
              'BraTS19_TMC_15477_1']

if __name__ == '__main__':
    opt = parser.parse_args()
    print(torch.__version__)
    print(opt)
    path = '/home/dlachinov/brats2019/data/2019_cropped'
    output_path = '/home/dlachinov/brats2019/data/out'

    trainer = train.Trainer(name=opt.name, models_root=opt.models_path, rewrite=False, connect_tb=False)
    trainer.load_best()#_load('_epoch_78')
    trainer.model = trainer.model.module.cpu()
    # trainer.model.eval()
    trainer.state.cuda = False

    series = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    series.sort()

    print(series)

    for f in series_val:

        image, label, affine = loader_helper.read_multimodal(path, f, True)


        #data_crop = median_filter(data_crop, 3)
#=========================================
        old_shape_crop = image.shape[1:]
        new_shape_crop = tuple([loader_helper.closest_to_k(i, 16) for i in old_shape_crop])
        diff = np.array(new_shape_crop) - np.array(old_shape_crop)
        pad_left = diff // 2
        pad_right = diff - pad_left

        #new_data_crop = np.full(shape=new_shape_crop, fill_value=-1000., dtype=np.float32)
        #new_data_crop[:old_shape_crop[0], :old_shape_crop[1], :old_shape_crop[2]] = data_crop

        new_data_crop = np.pad(image, pad_width=((0,0),)+tuple([(pad_left[i], pad_right[i]) for i in range(3)]),
                           mode='reflect')



        mask = new_data_crop > 0
        num_voxels = np.sum(mask, axis=(1, 2, 3))

        mean = np.sum(new_data_crop, axis=(1, 2, 3)) / num_voxels
        mean2 = np.sum(new_data_crop ** 2, axis=(1, 2, 3)) / num_voxels

        std = np.sqrt(mean2 - mean * mean)

        new_data_crop = (new_data_crop- mean.reshape((new_data_crop.shape[0], 1, 1, 1))) / std.reshape(
            (new_data_crop.shape[0], 1, 1, 1))
        new_data_crop[~mask] = 0

        #new_data_crop = new_data_crop.transpose((2,1,0))

        new_data_crops = []

        new_data_crops.append(new_data_crop)
        #new_data_crops.append(new_data_crop[::-1,:,:].copy())
        #new_data_crops.append(new_data_crop[:,::-1,:].copy())
        #new_data_crops.append(new_data_crop[::-1,::-1,:].copy()*1.2)
        #new_data_crops.append(new_data_crop*1.2)
        ##new_data_crops.append(new_data_crop[:,::-1,:].copy()+0.1)
        ##new_data_crops.append(new_data_crop[:,:,::-1].copy()+0.1)
        ##new_data_crops.append(new_data_crop[:,::-1,::-1].copy()+0.1)

        # tta
        outputs = []
        for new_data_crop in new_data_crops:
            new_data_crop = torch.from_numpy(new_data_crop[None, :, :, :, :]).float()

            output = trainer.predict([[new_data_crop], ])

            output_full = output[0].cpu().detach().numpy()[0]
            #output_ds = output[1].cpu().detach().numpy()[0]
            #output_ds = zoom(output_ds, (1,4,4,4), order=1, mode='constant', cval=0)
            #output_crop = (output_ds + output_full)/2
            output_crop = output_full

            outputs.append(output_crop)

        #outputs[1] = outputs[1][:, ::-1, :, :].copy()
        ##outputs[5] = outputs[5][:, :, ::-1, :].copy()

        #outputs[2] = outputs[2][:, :, ::-1, :].copy()
        ##outputs[6] = outputs[6][:, :, :, ::-1].copy()

        #outputs[3] = outputs[3][:, ::-1, ::-1, :].copy()
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

        output_crop = np.argmax(output_crop,axis=0).astype(np.uint8)

        output_header = nii.Nifti1Image(output_crop, affine)

        nii.save(output_header, os.path.join(output_path,f+'.nii.gz'))


        print(f, output_crop.shape, output_crop.dtype)

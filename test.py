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
from skimage import morphology


parser = argparse.ArgumentParser(description="PyTorch BraTS2019")
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

def get_bbox(data):
    bboxes = np.stack([loader_helper.bbox3(d) for d in data],axis=0)
    return np.stack([np.min(bboxes[:,0],axis=0),np.max(bboxes[:,1],axis=0)],axis=0)

def reject_small_regions(connectivity, ratio=0.25):
    resulting_connectivity = connectivity.copy()
    unique, counts = np.unique(connectivity, return_counts=True)

    all_nonzero_clusters = np.prod(connectivity.shape) - np.max(counts)

    for i in range(unique.shape[0]):
        if counts[i] < ratio * all_nonzero_clusters:
            resulting_connectivity[resulting_connectivity == unique[i]] = 0

    return resulting_connectivity

if __name__ == '__main__':
    opt = parser.parse_args()
    print(torch.__version__)
    print(opt)
    path = '/home/dlachinov/brats2019/data/MICCAI_BraTS_2018_Data_Validation'
    output_path = '/home/dlachinov/brats2019/data/out'

    trainer = train.Trainer(name=opt.name, models_root=opt.models_path, rewrite=False, connect_tb=False)
    trainer.load_best()
    trainer.state.cuda = True

    series = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    series.sort()

    print(series)

    for f in series:

        image, label, affine = loader_helper.read_multimodal(data_path=path, series=f, read_annotation=False)

        bbox = get_bbox(image)


        image_crop = image[:,bbox[0,0]:bbox[1,0],bbox[0,1]:bbox[1,1],bbox[0,2]:bbox[1,2]]


        #data_crop = median_filter(data_crop, 3)
#=========================================
        old_shape_crop = image_crop.shape[1:]
        new_shape_crop = tuple([loader_helper.closest_to_k(i, 16) for i in old_shape_crop])
        diff = np.array(new_shape_crop) - np.array(old_shape_crop)
        pad_left = diff // 2
        pad_right = diff - pad_left

        new_data_crop = np.pad(image_crop, pad_width=((0,0),)+tuple([(pad_left[i], pad_right[i]) for i in range(3)]),
                           mode='constant', constant_values=0)



        mask = new_data_crop > 0
        num_voxels = np.sum(mask, axis=(1, 2, 3))

        mean = np.sum(new_data_crop / num_voxels[:,None,None,None], axis=(1, 2, 3))
        mean2 = np.sum(np.square(new_data_crop)/ num_voxels[:,None,None,None], axis=(1, 2, 3))


        std = np.sqrt(mean2 - mean * mean)

        new_data_crop = (new_data_crop- mean.reshape((new_data_crop.shape[0], 1, 1, 1))) / std.reshape(
            (new_data_crop.shape[0], 1, 1, 1))

        new_data_crops = []

        new_data_crops.append(new_data_crop)
        new_data_crops.append(new_data_crop[:,::-1,:,:].copy())
        new_data_crops.append(new_data_crop[:,:,::-1,:].copy())
        new_data_crops.append(new_data_crop[:,::-1,::-1,:].copy())

        # tta
        outputs = []
        for new_data_crop in new_data_crops:
            new_data_crop = torch.from_numpy(new_data_crop[None, :, :, :, :]).float()

            output = trainer.predict([[new_data_crop], ])

            output_full = output[0].cpu().detach().numpy()[0]
            output_crop = output_full

            outputs.append(output_crop)

        outputs[1] = outputs[1][:, ::-1, :, :].copy()
        outputs[2] = outputs[2][:, :, ::-1, :].copy()
        outputs[3] = outputs[3][:, ::-1, ::-1, :].copy()

        output_crop = sum(outputs) / len(outputs)

        output_crop = output_crop[:, pad_left[0]:-pad_right[0] or None, pad_left[1]:-pad_right[1] or None,
                 pad_left[2]:-pad_right[2] or None]

# ==========================
        output_crop = output_crop > 0.5

        wt = output_crop[0]
        tc = output_crop[1]
        et = output_crop[2]

        wt_vol = wt.sum()
        tc_vol = tc.sum()
        et_vol = et.sum()


        output_crop = np.zeros(shape = output_crop.shape[1:],dtype = np.uint8)
        output_crop[wt] = 2
        output_crop[tc] = 1
        if et_vol > 32:
            output_crop[et] = 4


        connected_regions = morphology.label(output_crop > 0)
        clusters = reject_small_regions(connected_regions, 0.1)
        output_crop[clusters == 0] = 0


        output = np.zeros(shape=image.shape[1:],dtype=np.uint8)
        output[bbox[0, 0]:bbox[1, 0], bbox[0, 1]:bbox[1, 1], bbox[0, 2]:bbox[1, 2]] = output_crop


        output_header = nii.Nifti1Image(output, affine)

        nii.save(output_header, os.path.join(output_path,f+'.nii.gz'))


        print(f, output_crop.shape, output_crop.dtype, wt_vol, tc_vol, et_vol)

import nibabel as nib
import numpy as np
import os
import torch


def read_nii(filename):
    image = nib.load(filename)
    return np.array(image.get_data())

def read_numpy(filename):
    return np.load(filename)

def read_nii_header(filename):
    return nib.load(filename)


def read_multimodal(data_path, series, annotation_path=None, read_annotation=True):
    suffixes = ['_t1.nii.gz', '_t1ce.nii.gz', '_t2.nii.gz', '_flair.nii.gz']

    affine = read_nii_header(os.path.join(data_path, series, series + suffixes[0])).affine
    files = [read_nii(os.path.join(data_path, series, series + s)) for s in suffixes]
    data = np.stack(files, axis=0).astype(np.float32)
    annotation = None
    if read_annotation:
        p = os.path.join(data_path, series, series + '_seg.nii.gz')
        if annotation_path is not None and not os.path.isfile(p):
            p = os.path.join(annotation_path, series + '.nii.gz')
        annotation = read_nii(p)
        annotation[annotation == 4] = 3

    return data, annotation, affine

def get_indices(position, center_shape, border):
    index = [p * c for p, c in zip(position, center_shape)]
    index_min = [i - b for i, b in zip(index, border)]
    index_max = [i + c + b for i, c, b in zip(index, center_shape, border)]

    return index_min, index_max


def copy(data, tile_shape, index_min, index_max):
    ret = torch.zeros(size=data.shape[:2]+tuple(tile_shape), dtype=torch.float)
    #ret = np.zeros(shape=data.shape[:2] + tuple(tile_shape), dtype=np.float)

    index_clamp_min = np.maximum(index_min, 0)
    index_clamp_max = np.minimum(index_max, data.shape[2:])

    diff_min = [min_c - min_i for min_c, min_i in zip(index_clamp_min, index_min)]
    diff_max = [t - (max_i - max_c) for t, max_c, max_i in zip(tile_shape, index_clamp_max, index_max)]

    # print(index_min_center, index_max_center, diff_min, diff_max)

    ret[:,:,diff_min[0]:diff_max[0], diff_min[1]:diff_max[1], diff_min[2]:diff_max[2]] = \
        data[:,:,index_clamp_min[0]:index_clamp_max[0], index_clamp_min[1]:index_clamp_max[1],
        index_clamp_min[2]:index_clamp_max[2]]

    return ret


def ravel_index(index, grid):
    i = 0
    prod = 1
    for j in reversed(range(len(grid))):
        i = i + prod * index[j]
        prod = prod * grid[j]

    return i


def unravel_index(index, grid):
    i = []
    prod = np.prod(grid)
    for j in range(len(grid)):
        prod = prod // grid[j]
        i.append(index // prod)
        index = index % prod

    return i


def copy_back(data, tile, center_shape, index_min, index_max, border):
    index_center_min = [i + b for i, b in zip(index_min, border)]
    index_center_max = [i - b for i, b in zip(index_max, border)]

    index_clamp_min = np.maximum(index_center_min, 0)
    index_clamp_max = np.minimum(index_center_max, data.shape[2:])

    diff_min = [t + min_c - min_i for t, min_c, min_i in zip(border, index_clamp_min, index_center_min)]
    diff_max = [b + t - (max_i - max_c) for b, t, max_c, max_i in
                zip(border, center_shape, index_clamp_max, index_center_max)]

    #print(index_clamp_min, index_clamp_max, diff_min, diff_max)

    data[:,:,index_clamp_min[0]:index_clamp_max[0], index_clamp_min[1]:index_clamp_max[1],
    index_clamp_min[2]:index_clamp_max[2]] = \
        tile[:,:,diff_min[0]:diff_max[0], diff_min[1]:diff_max[1], diff_min[2]:diff_max[2]]

def closest_to_k(n,k=8):
    if n % k == 0:
        return n
    else:
        return ((n // k) + 1)*k

def bbox3(img):
    """
    compute bounding box of the nonzero image pixels
    :param img: input image
    :return: bbox with shape (2,3) and contents [min,max]
    """
    rows = np.any(img, axis=1)
    rows = np.any(rows, axis=1)

    cols = np.any(img, axis=0)
    cols = np.any(cols, axis=1)

    slices = np.any(img, axis=0)
    slices = np.any(slices, axis=0)

    rows = np.where(rows)
    cols = np.where(cols)
    slices = np.where(slices)
    if (rows[0].shape[0] > 0):
        rmin, rmax = rows[0][[0, -1]]
        cmin, cmax = cols[0][[0, -1]]
        smin, smax = slices[0][[0, -1]]

        return np.array([[rmin, cmin, smin], [rmax, cmax, smax]])
    return np.array([[-1,-1,-1],[0,0,0]])


def labels_to_regions(one_hot_labels):
    return None

def regions_to_labels(one_hot_regions):
    return None
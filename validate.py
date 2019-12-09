import loader_helper
import os
import torch
import argparse
import metrics
import numpy as np


parser = argparse.ArgumentParser(description="PyTorch BraTS2019 Validate")
parser.add_argument("--data_path", default="", type=str, help="path to train data")
parser.add_argument("--predictions_path", default="", type=str, help="path to output data")

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

series_val19 = [
    'BraTS19_CBICA_ANV_1',
    'BraTS19_CBICA_BBG_1',
    'BraTS19_TMC_15477_1'
]



if __name__ == '__main__':
    opt = parser.parse_args()
    print(torch.__version__)
    print(opt)
    path_input = opt.data_path
    path_output = opt.predictions_path

    series = [f for f in os.listdir(path_input) if os.path.isdir(os.path.join(path_input, f))]
    series.sort()

    #series = series_val19

    dice = metrics.Dice(input_index=0)
    dicewt = metrics.DiceWT(input_index=0)

    sum = 0

    for f in series:

        image, label, affine = loader_helper.read_multimodal(path_input, f, True)

        predict = loader_helper.read_nii(os.path.join(path_output,f+'.nii.gz')).astype(np.uint8)
        predict[predict==4] = 3


        result = np.zeros(shape=(4))

        for i in range(1, 4):
            p = (predict== i).astype(np.float32)
            g = (label == i).astype(np.float32)

            numerator = (p * g).sum()
            denominator = (p + g).sum()

            r = 2 * numerator / denominator
            if np.isnan(r):
                r = 1

            result[i-1] = r


        p = (predict > 0).astype(np.float32)
        g = (label > 0).astype(np.float32)

        numerator = (p * g).sum()
        denominator = (p + g).sum()

        r = 2 * numerator / denominator
        if np.isnan(r):
            r = 1

        result[3] = r

        sum = sum + result
        print(f, str(result))

    mean = sum / len(series)

    print(mean)




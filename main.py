import numpy as np
import argparse
import train
from model import UNet,UNet_hardcoded
import random
import torch
import torch.backends.cudnn as cudnn
import dataloader
from torch.utils.data.dataloader import DataLoader
import torch.utils.data as Data
import torch.optim as optim
import loss
import metrics
import os
from sklearn.model_selection import train_test_split, KFold
import pickle
import weight_init
import gc

parser = argparse.ArgumentParser(description="PyTorch SegTHOR")
parser.add_argument("--batchSize", type=int, default=1, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=500, help="number of epochs to train for")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--train_path", default="", type=str, help="path to train data")
parser.add_argument("--name", default="test", type=str, help="experiment name")
parser.add_argument("--models_path", default="/models", type=str, help="path to models folder")
parser.add_argument("--splits", default=1, type=int, help="number of splits in CV")


def worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    print('worker id {} seed {}'.format(worker_id, seed))


def main():

    opt = parser.parse_args()
    print(opt)
    print(torch.__version__)
    opt.seed = 1337
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)


    series = [f for f in os.listdir(opt.train_path) if os.path.isdir(os.path.join(opt.train_path, f))]
    series.sort()


    print("===> Building model")
    layers = [1,1,1,1,1]
    number_of_channels=[int(8*2**i) for i in range(1,1+len(layers))]#[32,128,256,512,1024]
    model = UNet(depth=len(layers), encoder_layers=layers, number_of_channels=number_of_channels, number_of_outputs=4)
    model.apply(weight_init.weight_init)
    model = torch.nn.DataParallel(module=model, device_ids=[0])

    trainer = train.Trainer(model=model, name=opt.name, models_root=opt.models_path, rewrite=True)
    trainer.cuda()
        
    gc.collect()

    opt.seed = 1337  # random.randint(1, 10000)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    print('Train data:', opt.train_path)


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

    series_train = [f for f in series if f not in series_val]
    print('Train {}'.format(series_train))
    print('Val {}'.format(series_val))

    train_set = dataloader.SimpleReader(path=opt.train_path,patch_size=(96, 128, 96), series=series_train, multiplier=16, patches_from_single_image=1)
    val_set = dataloader.FullReader(path=opt.train_path,series=series_val)

    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads,
                                      batch_size=opt.batchSize, shuffle=True, drop_last=True, worker_init_fn=worker_init_fn)

    batch_sampler = Data.BatchSampler(
        sampler=Data.SequentialSampler(val_set),
        batch_size=1,
        drop_last=True
    )

    evaluation_data_loader = DataLoader(dataset=val_set, num_workers=0,
                                    batch_sampler=batch_sampler)

    criterion = [loss.Dice_loss_joint(index=0,priority=1).cuda(),
                 loss.CE_Loss(index=0).cuda(),
                    ]
    print("===> Building model")

    print("===> Training")

    trainer.train(criterion=criterion,
                  optimizer=optim.Adam,
                  optimizer_params={"lr":1e-3,
                                    "weight_decay":1e-6,
                                    #"nesterov":True,
                                    #"momentum":0.9
                                    "amsgrad":True,
                                    },
                  scheduler=torch.optim.lr_scheduler.MultiStepLR,
                  scheduler_params={"milestones":[18000, 32000, 64000],
                                    "gamma":0.2,
                                    #"T_max":3000,
                                    #"eta_min":1e-3

                  },
                  training_data_loader=training_data_loader,
                  evaluation_data_loader=evaluation_data_loader,
                  split_into_tiles=False,
                  pretrained_weights=None,
                  train_metrics=[ metrics.Dice(name='Dice', input_index=0, target_index=0, classes=4),\
                                  metrics.DiceWT(name='DiceWT', input_index=0, target_index=0)
                                 ],
                  val_metrics=[metrics.Dice(name='Dice', input_index=0, target_index=0,classes=4),
                               metrics.DiceWT(name='DiceWT', input_index=0, target_index=0),
                               metrics.Hausdorff_ITK(name='Hausdorff_ITK', input_index=0, target_index=0,classes=4),
                               metrics.Hausdorff_ITKWT(name='Hausdorff_ITKWT', input_index=0, target_index=0)
                               ],
                  track_metric='Dice',
                  epoches=opt.nEpochs,
                  default_val=np.array([0,0,0,0,0]),
                  comparator=lambda x, y: np.min(x)+np.mean(x) > np.min(y)+np.mean(y),
                  eval_cpu=True
                  )


if __name__ == "__main__":
    #try:
    #    torch.multiprocessing.set_start_method('spawn')
    #except RuntimeError:
    #    pass
    main()

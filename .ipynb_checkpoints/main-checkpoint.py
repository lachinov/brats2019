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
parser.add_argument("--validation_path", default="", type=str, help="path to folder with val images")
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
    opt.seed = 1337  # random.randint(1, 10000)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    kf = KFold(n_splits=opt.splits)
    splits = []
    evaluation_metrics = []

    print("CV {} splits".format(kf.get_n_splits()))

    series = [f for f in os.listdir(opt.validation_path) if os.path.isdir(os.path.join(opt.validation_path, f))]
    series.sort()

    for idx, (train_index, test_index) in enumerate(kf.split(series)):

        print(train_index, test_index)

        print("===> Building model")
        layers = [2,2,3,4]
        model = UNet(depth=len(layers), encoder_layers=layers, number_of_channels=16, number_of_outputs=5).cuda()
        model.apply(weight_init.weight_init)
        model = torch.nn.DataParallel(module=model, device_ids=[0,1,2])

        trainer = train.Trainer(model=model, name=opt.name+str(idx), models_root=opt.models_path, rewrite=False)
        
        gc.collect()

        opt.seed = 1337  # random.randint(1, 10000)
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)

        cudnn.benchmark = True

        print("===> Loading datasets")
        print('Train data:', opt.train_path)


        series_train = [series[i] for i in train_index]
        series_val = [series[i] for i in test_index]
        print('Train {}'.format(series_train))
        print('Val {}'.format(series_val))

        #(176,96,128)
        #(16 * 17, 16 * 11, 16 * 6)
        #(272,176,96)
        #max is 255,176,130
        train_set = dataloader.SimpleReader(path=opt.train_path,patch_size=(16 * 15, 16 * 10, 16 * 6), series=series_train, multiplier=500, patches_from_single_image=1)
        val_set = dataloader.FullReader(path=opt.validation_path,series=series_val)#series_val
        #val_set = dataloader.SimpleReader(path=opt.validation_path,patch_size=(128,128,128), series=series_val, multiplier=5, patches_from_single_image=2)

        training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads,
                                      batch_size=opt.batchSize, shuffle=True, drop_last=True, worker_init_fn=worker_init_fn)

        batch_sampler = Data.BatchSampler(
            sampler=Data.SequentialSampler(val_set),
            batch_size=1,
            drop_last=True
        )

        evaluation_data_loader = DataLoader(dataset=val_set, num_workers=0,
                                        batch_sampler=batch_sampler)

        criterion = [loss.Dice_loss_joint(index=0,priority=1),
                     #loss.MSE_Loss_masked(index_mse=1, index_mask=0, priority=0.1),
                     loss.Dice_loss_joint(index=1,priority=0.5),
                     #loss.MSE_Loss_masked(index_mse=3, index_mask=2, priority=0.1),
                     #loss.CE_Loss(index=0,border=0),
                     #loss.GDL_joint(index=1, priority=0.2),
                     #loss.MSE_Loss(index=1,priority=0.1),
                     loss.BCE_Loss(label_index=0,bg_weight=1e-2).cuda(),
                     #loss.Dice_loss_joint(index=0, mask=None, border=8).cuda(),
                     #loss.BCE_Loss(label_index=2,mask=None,bg_weight=0.01).cuda(),
                     #loss.Dice_loss_joint(index=1,mask=None, border=1, priority=0.5).cuda(),# loss.CE_Loss(index=1,border=2).cuda(),
                     #loss.Dice1D(label_index=3).cuda(), #loss.BCE_Loss(label_index=1),
                     #loss.Dice1D(label_index=3).cuda(), loss.BCE_Loss(label_index=3)
                     #loss.sens_loss_joint(index=3).cuda(),
                    ]
        print("===> Building model")

        print("===> Training")

        trainer.train(criterion=criterion,
                  optimizer=optim.SGD,
                  optimizer_params={"lr":2e-1,
                                    "weight_decay":1e-6,
                                    #"nesterov":True,
                                    "momentum":0.9
                                    #   "amsgrad":True,
                                    },
                  scheduler=torch.optim.lr_scheduler.MultiStepLR,
                  scheduler_params={"milestones":[32000, 48000, 54000],
                                    "gamma":0.5,
                                    #"T_max":3000,
                                    #"eta_min":1e-3

                  },
                  training_data_loader=training_data_loader,
                  evaluation_data_loader=evaluation_data_loader,
                  split_into_tiles=False,
                  pretrained_weights=None,
                  train_metrics=[   #metrics.RMSE_masked(name='RMSE_masked', input_index=1,target_index=1,target_index_mask=0),
                                    #metrics.Dice1D(name='DiceA1D', input_index=3, target_index=3, classes=1),
                                    metrics.Dice(name='Dice', input_index=0, target_index=0),
                                    metrics.Dice(name='DiceDS', input_index=1, target_index=1),
                                 ],
                  val_metrics=[
                                    #metrics.RMSE_masked(name='RMSE_masked', input_index=1, target_index=1,target_index_mask=0),
                                #metrics.Dice1D(name='DiceA1D', input_index=3, target_index=3, classes=1),
                                metrics.Dice(name='Dice', input_index=0, target_index=0),
                                metrics.Dice(name='DiceDS', input_index=1, target_index=1),
                                metrics.Hausdorff_ITK(name='Hausdorff_ITK', input_index=0, target_index=0)
                               ],
                  track_metric='Dice',
                  epoches=opt.nEpochs,
                  default_val=np.array([0,0,0,0,0]),
                  comparator=lambda x, y: np.min(x)+np.mean(x) > np.min(y)+np.mean(y),
                  eval_cpu=True
                  )

        evaluation_metrics.append(trainer.state.best_val)
        splits.append((series_train, series_val))


    avg_val = 0

    for i in evaluation_metrics:
        avg_val = avg_val + i

    print('Average val {}'.format(avg_val/len(evaluation_metrics)))

    pickle.dump(evaluation_metrics, open(opt.name+'_eval.p', 'wb'))
    pickle.dump(splits, open(opt.name+'_splits.p','wb'))

if __name__ == "__main__":
    #try:
    #    torch.multiprocessing.set_start_method('spawn')
    #except RuntimeError:
    #    pass
    main()

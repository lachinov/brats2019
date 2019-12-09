import os
import shutil
import pickle
import torch
from torch.optim import *
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import numpy as np
import time
import loader_helper
import metrics
import gc

class TrainingState(object):
    def __init__(self):
        self.epoch = 0
        self.train_metric = dict()
        self.val_metric = dict()
        # number of processed batches
        self.global_step = 0
        self.best_val = 0
        self.optimizer_state = None
        self.cuda = True


class Trainer(object):
    def __init__(self, name, models_root, model=None, rewrite=False, connect_tb = True):

        self.model = model

        assert (isinstance(self.model, (list, tuple, torch.nn.Module)) or self.model is None)

        self.name = name
        self.models_root = models_root
        self.model_path = os.path.join(models_root, self.name)
        self.logs_path = os.path.join(self.model_path, 'logs')

        self.state = TrainingState()
        self.resume_training = False

        if os.path.exists(self.model_path):
            if rewrite:
                shutil.rmtree(self.model_path)
            else:
                self.resume_training = True

        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
            os.mkdir(self.logs_path)

        if connect_tb:
            self.tb_writer = SummaryWriter(logdir=self.logs_path)

    def cuda(self):
        if self.model is not None:
            self.model.cuda()
        self.state.cuda = True

    def train(self, criterion, optimizer, optimizer_params, scheduler, scheduler_params, training_data_loader,
              evaluation_data_loader, split_into_tiles, pretrained_weights, train_metrics, val_metrics,
              track_metric, epoches, default_val, comparator, eval_cpu, continue_form_pretraining):

        self.eval_cpu = eval_cpu

        assert (isinstance(criterion, (tuple, list, torch.nn.modules.loss._Loss)))

        # TODO: custom initializer here

        # load weights if any
        if self.resume_training:
            # load training and continue
            self.load_latest()
            #self._load('_epoch_65')
        elif pretrained_weights is not None:
            # load dictionary only
            self.model.load_state_dict(pretrained_weights)
        elif continue_form_pretraining:
            print('Continue from pretraining')
        else:
            self.state.best_val = default_val

        if isinstance(optimizer, type):
            optimizer = optimizer(params=self.model.parameters(), **optimizer_params)

        if scheduler is not None:
            if isinstance(scheduler, type):
                scheduler = scheduler(optimizer=optimizer, **scheduler_params)

        assert (isinstance(optimizer, torch.optim.Optimizer))
        assert (isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler) or scheduler is None)

        if self.state.optimizer_state is not None and not continue_form_pretraining:
            optimizer.load_state_dict(self.state.optimizer_state)
            print('Loaded optimizer state')

        # prepare dicts for metrics
        if not self.state.train_metric:
            for m in train_metrics:
                self.state.train_metric[m.name] = []

            for m in val_metrics:
                self.state.val_metric[m.name] = []

        gc.collect()
        
        # training loop
        start_epoch = self.state.epoch
        for i in range(start_epoch, epoches):
            tic = time.time()

            #if scheduler is not None:
            #    scheduler.step()

            self.state.global_step = self._train_one_epoch(criterion, optimizer, training_data_loader, train_metrics,
                                                           self.state.train_metric, i, self.state.global_step, scheduler)

            self._evaluate_and_save(evaluation_data_loader, split_into_tiles, val_metrics, track_metric, self.state.val_metric, i,
                                    comparator)

            tac = time.time()
            print('Epoch %d, time %s \n' % (i, tac - tic))

            self._save(suffix='_epoch_' + str(self.state.epoch))
            self._save(suffix='last_model')
            self.state.epoch = self.state.epoch + 1

        np.random.seed(np.random.get_state()[1][0] + 16)

    def predict(self, batch):
        self.model.eval()

        if self.state.cuda:
            self.model.cuda()

        with torch.no_grad():
            assert (isinstance(batch[0], list))
            data = batch[0]

            if self.state.cuda:
                data = [d.cuda() for d in data]

            output = self.model(data)
        return output

    def predict_tiled(self, batch, output_shape):
        # TODO: this is a workaround to support tiling for only signle input
        #       add tiling for selected inputs ( not just the 0th one)
        input = batch[0][0]
        output = torch.zeros(output_shape)

        if self.state.cuda:
            self.model.cuda()

        tile_shape = (192, 192, 192)
        center_shape = (48, 48, 48)
        border = (72, 72, 72)

        grid = [int(np.ceil(j / i)) for i, j in zip(center_shape, input.shape[2:])]

        for i in range(grid[0]):
            for j in range(grid[1]):
                for k in range(grid[2]):
                    index_min, index_max = loader_helper.get_indices(position=(i, j, k), center_shape=center_shape,
                                                                     border=border)
                    tile = loader_helper.copy(data=input, tile_shape=tile_shape, index_min=index_min,
                                              index_max=index_max)

                    if self.state.cuda:
                        tile = tile.cuda()

                    out = self.model([tile])[0].detach().cpu()

                    loader_helper.copy_back(data=output, tile=out, center_shape=center_shape, index_min=index_min,
                                            index_max=index_max, border=border)
        output = [output]
        return output

    def _train_one_epoch(self, criterion, optimizer, training_data_loader, train_metrics, train_metrics_results, epoch,
                         global_step, scheduler):

        aggregate_batches = 1
        for m in train_metrics:
            m.reset()

        if self.state.cuda:
            self.model.cuda()

        self.model.train()

        optimizer.zero_grad()
        for idx, batch in enumerate(training_data_loader):

            assert (isinstance(batch[0], list) and isinstance(batch[1], list))
            data = [Variable(b) for b in batch[0]]
            target = [Variable(b, requires_grad=False) for b in batch[1]]

            if self.state.cuda:
                data = [d.cuda() for d in data]
                target = [t.cuda() for t in target]

            output = self.model(data)

            if isinstance(criterion, (tuple, list)):
                loss_val = [c(output, target) for c in criterion]
                loss = sum(loss_val) / (len(loss_val))
            else:
                loss_val = criterion(output, target)
                loss = loss_val

            loss.backward()

            if (idx+1)%aggregate_batches == 0:

                #for name, param in self.model.named_parameters():
                #    self.tb_writer.add_scalar('misc/grad-max-{}'.format(name), torch.max(torch.abs(param.grad)).cpu().numpy(), global_step)

                #for param in self.model.parameters():
                #    param.grad.data = torch.clamp(param.grad.data, min=-1.0,max=1.0)

                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

            for m in train_metrics:
                m.update(output, target)

            for idx, l in enumerate(loss_val):
                self.tb_writer.add_scalar('loss/loss-{}'.format(idx), l.item(), global_step)

            for idx, param_group in enumerate(optimizer.param_groups):
                self.tb_writer.add_scalar('misc/lr-{}'.format(idx), param_group['lr'], global_step)

            global_step = global_step + 1

        for m in train_metrics:
            train_metrics_results[m.name].append(m.get())
            metrics.print_metrics(self.tb_writer, m, 'train/', epoch)

        self.state.optimizer_state = optimizer.state_dict()
        return global_step

    def _evaluate_and_save(self, evaluation_data_loader, split_into_tiles, val_metrics, track_metric, val_metrics_results, epoch,
                           comparator):

        for m in val_metrics:
            m.reset()

        self.model.eval()

        for batch in evaluation_data_loader:
            gc.collect()
            #torch.cuda.empty_cache()

            assert (isinstance(batch[0], list) and isinstance(batch[1], list))

            data = batch[0]
            target = batch[1]

            if split_into_tiles and not self.eval_cpu:

                #TODO: this is a workaround to support tiling for only signle input
                #       add tiling for selected inputs ( not just the 0th one)
                output = torch.zeros_like(batch[1][0])

                input = batch[0][0]

                tile_shape = (192, 192, 192)
                center_shape = (48, 48, 48)
                border = (72, 72, 72)

                grid = [int(np.ceil(j / i)) for i, j in zip(center_shape, input.shape[2:])]

                for i in range(grid[0]):
                    for j in range(grid[1]):
                        for k in range(grid[2]):
                            index_min, index_max = loader_helper.get_indices(position=(i, j, k), center_shape=center_shape, border=border)
                            tile = loader_helper.copy(data=input, tile_shape=tile_shape, index_min=index_min, index_max=index_max)

                            if self.state.cuda:
                                tile = tile.cuda()
                            with torch.no_grad():
                                out = self.model([tile])[0].detach().cpu()

                            loader_helper.copy_back(data=output,tile=out,center_shape=center_shape,index_min=index_min,index_max=index_max,border=border)

                output = [output]

            elif self.eval_cpu:
                tmp_model = self.model.module.cpu()
                tmp_model.eval()
                with torch.no_grad():
                    output = tmp_model(data)

            else:
                with torch.no_grad():
                    if self.state.cuda:
                        data = [d.cuda() for d in data]
                        target = [t.cuda() for t in target]

                    output = self.model(data)


            for m in val_metrics:
                m.update(target, output)

        val = 0.0
        for m in val_metrics:
            if m.name == track_metric:
                val = m.get()

            metrics.print_metrics(self.tb_writer, m, 'val/', epoch)
            val_metrics_results[m.name].append(m.get())

        if comparator(val, self.state.best_val):
            self.state.best_val = val
            self._save(suffix='best_model')
            print('model saved')

    def _save(self, suffix):
        s = {'state': self.state,
             'model': self.model}

        torch.save(s, os.path.join(self.model_path, self.name + suffix + '.pth'))

    def _load(self, suffix):
        print('loading model %s'%suffix)
        s = torch.load(os.path.join(self.model_path, self.name + suffix + '.pth'), map_location=torch.device('cpu'))
        self.state = s['state']
        if self.model is None:
            self.model = s['model']
        else:
            self.model.load_state_dict(s['model'].state_dict())

    def load_latest(self):
        self._load('last_model')

    def load_best(self):
        self._load('best_model')

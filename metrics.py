import torch
import numpy as np
import math
import SimpleITK as sitk

class Metrics(object):
    def __init__(self,name):
        self.name = name
        self.accumulator = 0.0
        self.samples = 0.0

    def update(self, ground, predict):
        self.samples = self.samples + 1

    def get(self):
        return self.accumulator / self.samples

    def reset(self):
        self.accumulator = 0.0
        self.samples = 0.0

class Dice1D(Metrics):
    def __init__(self, name='Dice1D', input_index=0,target_index=0, classes=4):
        super(Dice1D, self).__init__(name)
        self.input_index=input_index
        self.target_index=target_index
        self.classes = classes

    def update(self, ground, predict):
        pred = predict[self.input_index].detach()#.cpu().detach().numpy()
        gr = ground[self.target_index].detach()#.cpu().detach().numpy()

        assert (gr.shape == pred.shape)

        pred = (pred > 0.5).float()
        gr = (gr > 0.5).float()
        
        N = gr.shape[0]

        result = np.zeros(shape = (pred.shape[0],self.classes))

        for i in range(0, self.classes):
            p = pred[:,i].float().view(N,-1)
            g = gr[:,i].float().view(N,-1)
            #print(p.max(), g.max())
            r = 2 * (p * g).sum(dim=(1))/((p+g).sum(dim=(1))+1e-6)
            #print(r.shape)
            result[:,i] = r.cpu().numpy()

        self.accumulator = self.accumulator + result.mean(axis=0)

        self.samples += 1

class RMSE(Metrics):
    def __init__(self, name='RMSE', input_index=0,target_index=0):
        super(RMSE, self).__init__(name)
        self.input_index=input_index
        self.target_index=target_index

    def update(self, ground, predict):
        pred = predict[self.input_index].detach()#.cpu().detach().numpy()
        gr = ground[self.target_index].detach()#.cpu().detach().numpy()

        assert (gr.shape == pred.shape)

        pred = pred.view(-1)
        gr = gr.view(-1)

        result = (pred - gr)**2

        self.accumulator = self.accumulator + torch.sqrt(result.mean()).cpu().numpy()

        self.samples += 1

class RMSE_masked(Metrics):
    def __init__(self, name='RMSE_masked', input_index=0,target_index=0, target_index_mask=0):
        super(RMSE_masked, self).__init__(name)
        self.input_index=input_index
        self.target_index=target_index
        self.target_index_mask=target_index_mask

    def update(self, ground, predict):
        pred = predict[self.input_index].detach()#.cpu().detach().numpy()
        gr = ground[self.target_index].detach()#.cpu().detach().numpy()
        mask = ground[self.target_index_mask].detach()
        assert (gr.shape == pred.shape)

        #pred = x[self.index_mse]
        #gt = y[self.index_mse]

        mask = torch.unsqueeze((torch.sum(mask, dim=(2, 3)) > 0).float(), dim=3)[:,:2]

        sum = mask * (pred - gr) ** 2

        mse = (torch.sum(sum) / (torch.sum(mask)+1e-8))

        self.accumulator = self.accumulator + torch.sqrt(mse.mean()).cpu().numpy()

        self.samples += 1

class Dice(Metrics):
    def __init__(self, name='Dice', input_index=0,target_index=0, classes=4):
        super(Dice, self).__init__(name)
        self.input_index=input_index
        self.target_index=target_index
        self.classes = classes

    def update(self, ground, predict):
        pred = predict[self.input_index].detach()
        gr = ground[self.target_index].detach()

        assert (gr.shape == pred.shape)

        pred = pred > 0.5#torch.argmax(pred, dim=1).long().view(pred.size(0),-1)
        gr = gr > 0.5#torch.argmax(gr, dim=1).long().view(gr.size(0),-1)

        result = np.zeros(shape = (pred.shape[0],self.classes-1))

        for i in range(0, self.classes-1):
            p = pred[:,i].float().view(pred.size(0),-1)#(pred == i).float()
            g = gr[:,i].float().view(pred.size(0),-1)#(gr == i).float()

            numerator = (p * g).sum(dim=1).cpu().numpy()
            denominator = (p + g).sum(dim=1).cpu().numpy()

            r = 2 * numerator / denominator
            r[np.isnan(r)] = 1

            result[:,i] = r

        self.accumulator = self.accumulator + result.mean(axis=0)

        self.samples += 1

class DiceWT(Metrics):
    def __init__(self, name='Dice_WT', input_index=0,target_index=0):
        super(DiceWT, self).__init__(name)
        self.input_index=input_index
        self.target_index=target_index

    def update(self, ground, predict):
        pred = predict[self.input_index].detach()#.cpu().detach().numpy()
        gr = ground[self.target_index].detach()#.cpu().detach().numpy()

        #print(gr.shape, pred.shape)
        assert (gr.shape == pred.shape)

        pred = (torch.argmax(pred, dim=1)>0).float()#((torch.argmax(pred, dim=1) + 1) * (torch.max(pred, dim=1)[0] > 0.5).long()).long()#torch.argmax(pred, dim=1).long()
        gr = (torch.argmax(gr, dim=1)>0).float()#((torch.argmax(gr, dim=1) + 1) * (torch.max(gr, dim=1)[0] > 0.5).long()).long()#torch.argmax(gr, dim=1).long()

        r = 2 * (pred * gr).sum(dim=(1,2,3))/((pred+gr).sum(dim=(1,2,3))+1e-6)

        self.accumulator = self.accumulator + r.mean(dim=0).cpu().numpy()

        self.samples += 1

class Dice_ITK(Metrics):
    def __init__(self, name='Dice_ITK', input_index=0,target_index=0, classes=5):
        super(Dice_ITK, self).__init__(name)
        self.input_index=input_index
        self.target_index=target_index
        self.classes = classes
        self.overelap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()

    def update(self, ground, predict):
        pred = predict[self.input_index].detach()#.cpu().detach().numpy()
        gr = ground[self.target_index].detach()#.cpu().detach().numpy()

        assert (gr.shape == pred.shape)

        pred = torch.argmax(pred, dim=1).cpu().numpy()
        gr = torch.argmax(gr, dim=1).cpu().numpy()

        result = np.zeros(shape = (pred.shape[0],self.classes-1))
        for n in range(pred.shape[0]):
            for i in range(1, self.classes):
                p = (pred[n] == i).astype(np.uint8)
                g = (gr[n] == i).astype(np.uint8)
                self.overelap_measures_filter.Execute(sitk.GetImageFromArray(g), sitk.GetImageFromArray(p))

                result[n,i-1] = self.overelap_measures_filter.GetDiceCoefficient()

        self.accumulator = self.accumulator + result.mean(axis=0)

        self.samples += 1


class Hausdorff_ITK(Metrics):
    def __init__(self, name='Hausdorff_ITK', input_index=0,target_index=0, classes=5):
        super(Hausdorff_ITK, self).__init__(name)
        self.input_index=input_index
        self.target_index=target_index
        self.classes = classes
        self.hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

    def update(self, ground, predict):
        pred = predict[self.input_index].detach()#.cpu().detach().numpy()
        gr = ground[self.target_index].detach()#.cpu().detach().numpy()

        assert (gr.shape == pred.shape)

        #pred = torch.argmax(pred, dim=1).long().cpu().numpy()
        #gr = torch.argmax(gr, dim=1).long().cpu().numpy()

        pred = (pred > 0.5).long().cpu().numpy()#torch.argmax(pred, dim=1).long().cpu().numpy()#((torch.argmax(pred, dim=1) + 1) * (torch.max(pred, dim=1)[0] > 0.5).long()).long().cpu().numpy()#torch.argmax(pred, dim=1).long()
        gr = (gr > 0.5).long().cpu().numpy()#torch.argmax(gr, dim=1).long().cpu().numpy()#((torch.argmax(gr, dim=1) + 1) * (torch.max(gr, dim=1)[0] > 0.5).long()).long().cpu().numpy()#torch.argmax(gr, dim=1).long()

        result = np.zeros(shape = (pred.shape[0],self.classes-1))

        for n in range(pred.shape[0]):
            for i in range(0, self.classes-1):
                p = pred[n,i].astype(np.uint8)
                g = gr[n,i].astype(np.uint8)

                if p.sum() == 0 and g.sum() == 0:
                    result[n,i-1] = 0
                    continue

                r = 1e+6
                try:
                    self.hausdorff_distance_filter.Execute(sitk.GetImageFromArray(g), sitk.GetImageFromArray(p))
                    r = self.hausdorff_distance_filter.GetHausdorffDistance()
                except RuntimeError:
                    print("Hausdorff_ITK:RuntimeError")
                    pass

                result[n,i] = r

        self.accumulator = self.accumulator + result.mean(axis=0)

        self.samples += 1

class Hausdorff_ITKWT(Metrics):
    def __init__(self, name='Hausdorff_ITKWT', input_index=0,target_index=0):
        super(Hausdorff_ITKWT, self).__init__(name)
        self.input_index=input_index
        self.target_index=target_index
        self.hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

    def update(self, ground, predict):
        pred = predict[self.input_index].detach()#.cpu().detach().numpy()
        gr = ground[self.target_index].detach()#.cpu().detach().numpy()

        assert (gr.shape == pred.shape)

        #pred = torch.argmax(pred, dim=1).long().cpu().numpy()
        #gr = torch.argmax(gr, dim=1).long().cpu().numpy()

        pred = (torch.argmax(pred, dim=1)>0).long().cpu().numpy()#((torch.argmax(pred, dim=1) + 1) * (torch.max(pred, dim=1)[0] > 0.5).long()).long().cpu().numpy()#torch.argmax(pred, dim=1).long()
        gr = (torch.argmax(gr, dim=1)>0).long().cpu().numpy()#((torch.argmax(gr, dim=1) + 1) * (torch.max(gr, dim=1)[0] > 0.5).long()).long().cpu().numpy()#torch.argmax(gr, dim=1).long()

        result = np.zeros(shape = (pred.shape[0]))

        for n in range(pred.shape[0]):
            p = pred[n].astype(np.uint8)
            g = gr[n].astype(np.uint8)


            r = 1e+6
            try:
                self.hausdorff_distance_filter.Execute(sitk.GetImageFromArray(g), sitk.GetImageFromArray(p))
                r = self.hausdorff_distance_filter.GetHausdorffDistance()
            except RuntimeError:
                print("Hausdorff_ITK:RuntimeError")
                pass

            result[n] = r

        self.accumulator = self.accumulator + result.mean(axis=0)

        self.samples += 1

def print_metrics(writer, metric, prefix, epoch):
    if isinstance(metric.get(), np.ndarray):
        for i in range(metric.get().shape[0]):
            writer.add_scalar(prefix + metric.name+str(i), metric.get()[i], epoch)
    else:
        writer.add_scalar(prefix + metric.name, metric.get(), epoch)

    print('Epoch %d, %s %s %s' % (epoch, prefix, metric.name, metric.get()))

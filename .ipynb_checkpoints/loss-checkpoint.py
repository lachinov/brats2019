import torch
import torch.nn as nn

class LossWrapper(nn.modules.loss._Loss):
    def __init__(self, loss, input_index, target_index, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(LossWrapper, self).__init__(size_average, reduce, reduction)
        self.loss = loss
        self.target_index = target_index
        self.input_index = input_index

    def forward(self, input, target):
        return self.loss(input[self.input_index], target[self.target_index])

class MSE_Loss(nn.Module):
    def __init__(self, index = 0):
        super(MSE_Loss, self).__init__()
        self.index = index

    def forward(self, x, y):
        assert (x[self.index].shape == y[self.index].shape)

        pred = x[self.index].view(-1)
        gt = y[self.index].view(-1)

        loss = (pred - gt)**2

        return torch.mean(loss)


class CE_Loss(nn.Module):
    def __init__(self, index = 0, border = 32):
        super(CE_Loss, self).__init__()
        self.border = border
        self.index = index

    def forward(self, x, y):
        assert (x[self.index].shape == y[self.index].shape)
        upper_limit = [ s - self.border for s in x[0].shape[2:]]

        loss = y[self.index][:,:,self.border:upper_limit[0], self.border:upper_limit[1], self.border:upper_limit[2]]*\
               torch.log(x[self.index][:,:,self.border:upper_limit[0], self.border:upper_limit[1], self.border:upper_limit[2]]+1e-6)*\
               torch.Tensor([0.005, 1.]).view(1,2,1,1,1).cuda()

        return -torch.mean(loss)

class BCE_Loss(nn.Module):
    def __init__(self, label_index = 0, mask = None):
        super(BCE_Loss, self).__init__()
        self.label_index = label_index
        self.mask = mask

    def forward(self, x, y):
        assert (x[self.label_index].shape == y[self.label_index].shape)
        
        pred = x[self.label_index]
        gt = y[self.label_index]
        
        if self.mask is not None:
            m = (x[self.mask].detach() > 0.5).float()
            pred = pred * m
            gt = gt * m            

        loss = gt*torch.log(pred+1e-6) + \
        (1. - gt)* torch.log(1. - pred+1e-6)

        return -torch.mean(loss)

class Dice1D(nn.Module):
    def __init__(self, label_index = 0):
        super(Dice1D, self).__init__()
        self.label_index = label_index

    def forward(self, x, y):
        assert (x[self.label_index].shape == y[self.label_index].shape)

        shape = x[self.label_index].shape
        x = x[self.label_index].view(shape[0], shape[1], -1)
        y = y[self.label_index].view(shape[0], shape[1], -1)

        intersection = (x * y).sum(dim=(0,2)) + 1
        union = (x + y).sum(dim=(0,2)) + 2

        return 1.0-2.0*torch.mean(intersection / union)

class Dice_loss_joint(nn.Module):
    def __init__(self, index = 0, mask = None, border = 32, priority = 1):
        super(Dice_loss_joint, self).__init__()
        self.border = border
        self.index = index
        self.mask = mask
        self.priority = priority


    def forward(self, x, y):
        #print(x[self.index].shape, y[self.index].shape)
        assert (x[self.index].shape == y[self.index].shape)

        upper_limit = [ s - self.border for s in x[self.index].shape[2:]]

        pred = x[self.index][:,1:,self.border:upper_limit[0], self.border:upper_limit[1], self.border:upper_limit[2]]
        gt = y[self.index][:,1:,self.border:upper_limit[0], self.border:upper_limit[1], self.border:upper_limit[2]]
        
        if self.mask is not None:
            m = (x[self.mask][:,:,self.border:upper_limit[0], :, :].detach() > 0.5).float()
            pred = pred * m
            gt = gt * m

        intersection = (pred*gt).sum(dim=(0,2,3,4)) + 1
        union = (pred**2 + gt**2).sum(dim=(0,2,3,4)) + 2
        
        dice = 2.0*intersection / union
        
        #exp_loss = torch.pow(-torch.log(dice),0.3)
        #print(dice.cpu().detach())

        return self.priority*(1.0 - torch.mean(dice))


class sens_loss_joint(nn.Module):
    def __init__(self, index=0, priority=1):
        super(sens_loss_joint, self).__init__()
        self.index = index
        self.priority = priority

    def forward(self, x, y):
        # print(x[self.index].shape, y[self.index].shape)
        assert (x[self.index].shape == y[self.index].shape)

        pred = x[self.index]
        gt = y[self.index]

        intersection = (pred * gt).sum(dim=(0, 2, 3, 4)) + 1
        union = gt.sum(dim=(0, 2, 3, 4)) + 1

        loss = intersection / union

        # exp_loss = torch.pow(-torch.log(dice),0.3)
        # print(dice.cpu().detach())

        return self.priority * (1.0 - torch.mean(loss))

class Dice_loss_separate(nn.Module):
    def __init__(self, index = 0, mask = None, border = 32):
        super(Dice_loss_separate, self).__init__()
        self.border = border
        self.index = index
        self.mask = mask


    def forward(self, x, y):
        #print(x[self.index].shape, y[self.index].shape)
        assert (x[self.index].shape == y[self.index].shape)

        upper_limit = [ s - self.border for s in x[self.index].shape[2:]]
        
        pred = x[self.index][:,:,self.border:upper_limit[0], self.border:upper_limit[1], self.border:upper_limit[2]]
        gt = y[self.index][:,:,self.border:upper_limit[0], self.border:upper_limit[1], self.border:upper_limit[2]]
        
        if self.mask is not None:
            m = (x[self.mask][:,:,self.border:upper_limit[0], :, :].detach() > 0.5).float()
            pred = pred * m
            gt = gt * m

        intersection = (pred*gt).sum(dim=(2,3,4))
        union = (pred + gt).sum(dim=(2,3,4)) + 1e-6

        return 1.0-torch.mean(2.0*intersection / union)
import torch
import torch.nn as nn
import torch.nn.functional as F

class LossWrapper(nn.modules.loss._Loss):
    def __init__(self, loss, input_index, target_index, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(LossWrapper, self).__init__(size_average, reduce, reduction)
        self.loss = loss
        self.target_index = target_index
        self.input_index = input_index

    def forward(self, input, target):
        return self.loss(input[self.input_index], target[self.target_index])

class MSE_Loss(nn.Module):
    def __init__(self, index = 0, priority=1):
        super(MSE_Loss, self).__init__()
        self.index = index
        self.priority=priority

    def forward(self, x, y):
        assert (x[self.index].shape == y[self.index].shape)

        pred = x[self.index].view(-1)
        gt = y[self.index].view(-1)

        loss = (pred - gt)**2

        return torch.mean(loss)*self.priority

class MSE_Loss_masked(nn.Module):
    def __init__(self, index_mse = 1, index_mask=0, priority=1):
        super(MSE_Loss_masked, self).__init__()
        self.index_mse = index_mse
        self.index_mask = index_mask
        self.priority=priority

    def forward(self, x, y):
        #print(x[self.index_mse].shape, y[self.index_mse].shape)
        assert (x[self.index_mse].shape == y[self.index_mse].shape)

        pred = x[self.index_mse]
        gt = y[self.index_mse]

        mask = torch.unsqueeze((torch.sum(y[self.index_mask],dim=(2,3)) > 0).float(),dim=3)[:,:2]

        loss = mask*(pred - gt)**2

        return (torch.mean(loss)/(torch.sum(mask)+1e-8))*self.priority


class CE_Loss(nn.Module):
    def __init__(self, index = 0):
        super(CE_Loss, self).__init__()
        self.index = index

    def forward(self, x, y):
        assert (x[self.index].shape == y[self.index].shape)

        loss = y[self.index]*torch.log(x[self.index]+1e-6)

        return -torch.mean(loss)

class BCE_Loss(nn.Module):
    def __init__(self, index = 0, bg_weight=1):
        super(BCE_Loss, self).__init__()
        self.label_index = index
        self.bg_weight=bg_weight

    def forward(self, x, y):
        assert (x[self.label_index].shape == y[self.label_index].shape)
        
        pred = x[self.label_index]
        gt = y[self.label_index]

        loss = gt*torch.log(pred+1e-6) + \
        self.bg_weight*(1. - gt)* torch.log((1.+1e-6) - pred)

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
    def __init__(self, index = 0, priority = 1):
        super(Dice_loss_joint, self).__init__()
        self.index = index
        self.priority = priority


    def forward(self, x, y):
        #print(x[self.index].shape, y[self.index].shape)
        assert (x[self.index].shape == y[self.index].shape)
        N, C, H, W, D = x[self.index].shape

        pred = x[self.index].view(N, C, -1)
        gt = y[self.index].view(N, C, -1)


        intersection = (pred*gt).sum(dim=(0,2)) + 1e-6
        union = (pred**2 + gt).sum(dim=(0,2)) + 2e-6
        
        dice = 2.0*intersection / union
        
        #exp_loss = torch.pow(-torch.log(dice),0.3)
        #print(dice.cpu().detach())

        return self.priority*(1.0 - torch.mean(dice))


class GDL_joint(nn.Module):
    def __init__(self, index=0, priority=1):
        super(GDL_joint, self).__init__()
        self.index = index
        self.priority = priority

    def forward(self, x, y):
        # print(x[self.index].shape, y[self.index].shape)
        assert (x[self.index].shape == y[self.index].shape)

        N, C, H, W, D = x[self.index].shape

        pred = x[self.index].view(N, C, -1)[:,1:]
        gt = y[self.index].view(N, C, -1)[:,1:]

        w = 1 / gt.sum(dim=(0,2))

        intersection = w*((pred * gt).sum(dim=(0, 2)) + 1)
        union = w*((pred**2 + gt).sum(dim=(0, 2)) + 1)

        dice = 2.0 * intersection.sum() / union.sum()

        # exp_loss = torch.pow(-torch.log(dice),0.3)
        # print(dice.cpu().detach())

        return self.priority * (1.0 - torch.mean(dice))


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
    def __init__(self, index = 0, priority = 1):
        super(Dice_loss_separate, self).__init__()
        self.index = index
        self.priority = priority

    def forward(self, x, y):
        #print(x[self.index].shape, y[self.index].shape)
        assert (x[self.index].shape == y[self.index].shape)
        N, C, H, W, D = x[self.index].shape
        
        pred = x[self.index].view(N, C, -1)[:,1:]
        gt = y[self.index].view(N, C, -1)[:,1:]

        intersection = (pred*gt).sum(dim=2)
        union = (pred**2 + gt).sum(dim=2)

        dice = (2.0*intersection+1) / (union+1)

        return 1.0-torch.mean(dice)

import torch
import torch.nn as nn
import math
import gc
import torch.nn.functional as F

class Trilinear(nn.Module):
    def __init__(self, scale):
        super(Trilinear, self).__init__()
        self.scale = scale

    def forward(self, x):
        out = F.interpolate(x,scale_factor=self.scale,mode='trilinear')
        return out


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.att = nn.Sequential(
            nn.Conv3d(in_channels=channel, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)

        att = self.att(x)

        return x * att * y.expand_as(x)


class Mean(nn.Module):
    def __init__(self):
        super(Mean, self).__init__()

    def forward(self, x):

        N, C, D, H, W = x.shape

        #make sum equals to one
        s = torch.unsqueeze(torch.unsqueeze(torch.sum(x, dim=(2,3)),dim=2),dim=2) + 1e-8
        #print('Mean sum ', s)
        x = x / s

        d_indices = torch.arange(start=0, end=D, step=1).view(1,1,-1,1).float().to(x)
        h_indices = torch.arange(start=0, end=H, step=1).view(1,1,-1,1).float().to(x)
        #w_indices = torch.arange(start=0, end=W-1, step=1)

        d_mean = torch.sum(torch.sum(x, dim=3) * d_indices,dim=2)
        h_mean = torch.sum(torch.sum(x, dim=2) * h_indices,dim=2)
        #w_mean = torch.sum(torch.sum(x, dim=(2,3)) * w_indices,dim=2)

        return torch.cat([torch.unsqueeze(d_mean,dim=3), torch.unsqueeze(h_mean,dim=3)], dim=3)


class conv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(conv, self).__init__()

        #self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels,
        #                       kernel_size=(1, 1, 7), stride=(1,1,stride), padding=(0, 0, 3), bias=False, groups=1)
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=(3,3,3), stride=stride, padding=1, bias=False, groups=groups)

    def forward(self, x):
        out = x
        out = self.conv1(out)
        #out = self.conv2(out)
        return out

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample = None, conv_groups=1):
        super(Residual, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.downsample = downsample

        self.conv1 = conv(in_channels=in_channels, out_channels=out_channels, stride=stride)
        #self.pad1 = Pad(size=1)
        self.conv2 = conv(in_channels=out_channels, out_channels=out_channels,stride=1)
        #self.pad2 = Pad(size=1)
        self.relu1 = nn.LeakyReLU(1e-2,inplace=True)#nn.ReLU(inplace=True)
        self.relu2 = nn.LeakyReLU(1e-2,inplace=True)#nn.ReLU(inplace=True)
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)#nn.InstanceNorm3d(num_features=out_channels,affine=True)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)#nn.InstanceNorm3d(num_features=out_channels,affine=True)
        #self.se = SEBlock(channel=in_channels, reduction=4)

    def forward(self, x):

        if self.downsample is not None:
            x = self.downsample(x)

        out = x
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.relu1(out)
        #out = self.pad1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        #out = self.pad2(out)

        out = x + out

        return out


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample = None):
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.downsample = downsample

        self.conv1 = conv(in_channels=in_channels, out_channels=out_channels,stride=stride)
        self.conv2 = conv(in_channels=out_channels, out_channels=out_channels, stride=1)
        self.relu1 = nn.ReLU(inplace=True)#nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)#nn.ReLU(inplace=True)\

        self.norm1 = nn.BatchNorm3d(num_features=out_channels,affine=True,track_running_stats=True,momentum=0.5)#nn.GroupNorm(num_channels=in_channels,num_groups=norm_groups)#nn.InstanceNorm3d(num_features=out_channels)
        self.norm2 = nn.BatchNorm3d(num_features=out_channels,affine=True,track_running_stats=True,momentum=0.5)#nn.GroupNorm(num_channels=out_channels,num_groups=norm_groups)#nn.InstanceNorm3d(num_features=out_channels)

    def forward(self, x):

        if self.downsample is not None:
            x = self.downsample(x)

        out = x
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.relu1(out)


        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)

        return out

class ResNextBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample = None, width=4, compression=4):
        super(ResNextBottleneck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.downsample = downsample

        conv_groups = out_channels // (width*compression)

        self.conv_pre = nn.Conv3d(in_channels=in_channels, out_channels=out_channels//compression, kernel_size=1, stride=1,
                               padding=0, bias=False, groups=1)
        self.conv1 = conv(in_channels=out_channels//compression, out_channels=out_channels//compression,stride=stride, groups=conv_groups)
        #nn.Conv3d(in_channels=out_channels//compression, out_channels=out_channels//compression, kernel_size=3, stride=stride, padding=1, bias=False, groups=conv_groups)
        #self.pad1 = Pad(size=1)
        #self.conv2 = conv(in_channels=out_channels//compression, out_channels=out_channels//compression,stride=1)
        #nn.Conv3d(in_channels=out_channels//compression, out_channels=out_channels//compression, kernel_size=3, padding=1, bias=False, groups=conv_groups)
        #self.pad2 = Pad(size=1)
        self.relu1 = nn.LeakyReLU(1e-2,inplace=True)#nn.ReLU(inplace=True)
        self.relu2 = nn.LeakyReLU(1e-2,inplace=True)#nn.ReLU(inplace=True)
        self.relu3 = nn.LeakyReLU(1e-2,inplace=True)#nn.ReLU(inplace=True)
        self.norm1 = nn.GroupNorm(num_channels=out_channels//compression,num_groups=8)#nn.InstanceNorm3d(num_features=out_channels)
        self.norm2 = nn.GroupNorm(num_channels=out_channels//compression,num_groups=8)#nn.InstanceNorm3d(num_features=out_channels)
        self.norm3 = nn.GroupNorm(num_channels=out_channels,num_groups=8)#nn.InstanceNorm3d(num_features=out_channels)
        self.conv_post = nn.Conv3d(in_channels=out_channels // compression, out_channels=out_channels,
                               kernel_size=1, padding=0, bias=False, groups=1)
        self.se = SEBlock(channel=out_channels, reduction=4)

    def forward(self, x):

        if self.downsample is not None:
            x = self.downsample(x)

        out = x
        out = self.conv_pre(out)
        out = self.norm1(out)
        out = self.relu1(out)


        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu2(out)

        out = self.conv_post(out)
        out = self.norm3(out)

        out = x + out
        out = self.se(out)
        out = self.relu3(out)

        return out

class shuffle(nn.Module):
    def __init__(self, ratio):
        super(shuffle, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        batch_size, in_channels, d, h, w = x.shape
        out_channels = in_channels // (self.ratio*self.ratio*self.ratio)
        out = x.view(batch_size*out_channels, self.ratio,self.ratio,self.ratio,d,h,w)
        out = out.permute(0,4,1,5,2,6,3)

        return out.contiguous().view(batch_size, out_channels, d*self.ratio, h*self.ratio, w*self.ratio)


class re_shuffle(nn.Module):
    def __init__(self, ratio):
        super(re_shuffle, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        batch_size, in_channels, d, h, w = x.shape

        out_channels = in_channels * self.ratio * self.ratio * self.ratio
        out = x.view(batch_size*in_channels, d//self.ratio, self.ratio, h//self.ratio, self.ratio, w//self.ratio, self.ratio)
        out = out.permute(0,2,4,6,1,3,5)
        out = out.contiguous().view(batch_size, out_channels, d//self.ratio, h//self.ratio, w//self.ratio)
        return out


class DownsamplingPixelShuffle(nn.Module):
    def __init__(self, input_channels, output_channels,ratio=2):
        super(DownsamplingPixelShuffle, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv = nn.Conv3d(in_channels = input_channels*int(ratio**3), out_channels=output_channels, kernel_size=1, padding=0, bias=False)
        #self.pad = Pad(size=1)
        self.relu = nn.LeakyReLU(2e-2, inplace=True)
        self.shuffle = re_shuffle(ratio=ratio)

    def forward(self, x):
        #out = self.pad(x)
        out = self.shuffle(x)
        out = self.conv(out)
        return out

class UpsamplingPixelShuffle(nn.Module):
    def __init__(self, input_channels, output_channels,ratio=2):
        super(UpsamplingPixelShuffle, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv = nn.Conv3d(in_channels = input_channels, out_channels=output_channels*int(ratio**3), kernel_size=1, padding=0, bias=False)
        #self.pad = Pad(size=1)
        self.relu = nn.LeakyReLU(2e-2, inplace=True)
        self.shuffle = shuffle(ratio=ratio)

    def forward(self, x):
        #out = self.pad(x)
        out = self.conv(x)
        out = self.shuffle(out)
        return out

class Attention(nn.Module):
    def __init__(self, input_channels, attention_gates=1):
        super(Attention, self).__init__()
        self.input_channels = input_channels
        self.attention_gates = attention_gates
        self.conv_enc = nn.Conv3d(in_channels = input_channels, out_channels=input_channels, kernel_size=1, padding=0, bias=False)
        self.conv_dec = nn.Conv3d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, padding=0,
                                  bias=False)
        self.conv_output = nn.Conv3d(in_channels=input_channels, out_channels=attention_gates, kernel_size=1, padding=0,
                                  bias=False)

        self.relu = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()

        self.conv_last_channel_attention = nn.Conv1d(in_channels=input_channels, out_channels=attention_gates, kernel_size=5, padding=2,
                                     bias=False)


    def forward(self, from_encoder, from_decoder):
        N, C, D, H, W = from_decoder.shape


        enc = self.conv_enc(from_encoder)
        dec = self.conv_dec(from_decoder)

        feature_map = enc + dec
        out = self.relu(feature_map)
        out = self.conv_output(out)
        out = self.sigmoid(out)

        avg_pool = feature_map.mean(dim=(2,3))
        att_last_channel = self.conv_last_channel_attention(avg_pool).view(N, self.attention_gates,1,1,1,W)
        att_last_channel = self.sigmoid(att_last_channel)

        out = out.view(N, self.attention_gates, 1, D, H, W)
        dec = from_decoder.view(N, self.attention_gates, C // self.attention_gates, D, H, W)

        spatial_attention = dec * out
        attention = spatial_attention * att_last_channel

        return (attention).view(N, C, D, H, W)

class UNet(nn.Module):
    def __init__(self, depth, encoder_layers, decoder_layers, number_of_channels, number_of_outputs, block=Residual):
        super(UNet, self).__init__()
        print('UNet {}'.format(number_of_channels))

        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers

        self.number_of_channels = number_of_channels
        self.number_of_outputs = number_of_outputs
        self.depth = depth
        self.block = block

        self.conv_input = 0

        self.encoder_convs = nn.ModuleList()

        self.upsampling = nn.ModuleList()

        self.decoder_convs = nn.ModuleList()

        self.decoder_convs1x1 = nn.ModuleList()

        self.attention_convs = nn.ModuleList()

        self.upsampling_distance = nn.ModuleList()

        #self.padding1 = Pad(size=1)
        self.conv_input = nn.Conv3d(in_channels=4, out_channels=self.number_of_channels[0], kernel_size=(3,3,3), stride=1, padding=(1,1,1),
                                    bias=False)
        self.norm_input = nn.GroupNorm(num_groups=8, num_channels=self.number_of_channels[0])#nn.BatchNorm3d(num_features=self.number_of_channels[0],affine=True,track_running_stats=True,momentum=0.5)

        conv_first_list = []
        #conv_first_list.append(nn.Dropout3d(p=0.25))
        for i in range(self.encoder_layers[0]):
            conv_first_list.append(self.block(in_channels=self.number_of_channels[0],out_channels=self.number_of_channels[0],stride=1))

        self.conv_first = nn.Sequential(*conv_first_list)


        self.conv_output = nn.Conv3d(in_channels=self.number_of_channels[0],out_channels=self.number_of_outputs,kernel_size=3, stride=1,padding=1,bias=True,groups=1)

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU(1e-2,inplace=True)

        self.construct_dencoder_convs(depth=depth,number_of_channels=number_of_channels)
        self.construct_encoder_convs(depth=depth,number_of_channels=number_of_channels)
        self.construct_upsampling_convs(depth=depth,number_of_channels=number_of_channels)

    def _make_encoder_layer(self, in_channels, channels, blocks, stride=1, block=ResNextBottleneck):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv3d(in_channels=in_channels, out_channels=channels, kernel_size=2, stride=stride, bias=False)
            )

        layers = []
        layers.append(block(in_channels=channels, out_channels=channels, stride=1, downsample=downsample))

        for _ in range(1, blocks):
            layers.append(block(in_channels=channels,out_channels=channels,stride=1))


        return nn.Sequential(*layers)

    def construct_encoder_convs(self, depth, number_of_channels):
        for i in range(depth-1):
            conv = self._make_encoder_layer(in_channels=number_of_channels[i],channels=number_of_channels[i+1],blocks=self.encoder_layers[i+1],stride=2, block=self.block)
            self.encoder_convs.append(conv)

    def construct_dencoder_convs(self, depth, number_of_channels):
        for i in range(depth):

            block = self.block


            conv_list = []
            for j in range(self.decoder_layers[i]):
                conv_list.append(block(in_channels=number_of_channels[i], out_channels=number_of_channels[i], stride=1))

            conv = nn.Sequential(
                *conv_list
            )

            conv1x1 = nn.Conv3d(in_channels=2*number_of_channels[i], out_channels=number_of_channels[i], kernel_size=1, padding=0, bias=False)
            self.decoder_convs.append(conv)
            self.decoder_convs1x1.append(conv1x1)

    def construct_upsampling_convs(self, depth, number_of_channels):
        for i in range(depth-1):
            conv = nn.Sequential(
                Trilinear(scale=2),
                nn.Conv3d(in_channels=number_of_channels[i+1], out_channels=number_of_channels[i], kernel_size=1, stride=1, bias=False)
            )

            self.upsampling.append(conv)


    def forward(self, x):
        skip_connections = []
        gc.collect()
        input = x[0]

        conv = self.conv_input(input)
        conv = self.norm_input(conv)
        conv = self.conv_first(conv)
        
        for i in range(self.depth-1):
            skip_connections.append(conv)
            conv = self.encoder_convs[i](conv)

        for i in reversed(range(self.depth-1)):
            conv = self.upsampling[i](conv)
            conv = self.relu(conv)

            conc = torch.cat([skip_connections[i],conv],dim=1)
            conv = self.decoder_convs1x1[i](conc)
            conv = self.decoder_convs[i](conv)


        out_logits = self.conv_output(conv)

        out_logits = self.sigmoid(out_logits)

        return [out_logits]
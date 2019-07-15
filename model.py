import torch
import torch.nn as nn
import math
import gc
import torch.nn.functional as F

class LCNorm(nn.Module):
    def __init__(self):
        super(LCNorm, self).__init__()

    def forward(self, x):
        x2_mean = F.avg_pool3d(x * x, kernel_size=9, padding=4, stride=1)
        x_mean = F.avg_pool3d(x, kernel_size=9, padding=4, stride=1)
        x_std = torch.sqrt(x2_mean - x_mean * x_mean + 0.03)
        x = (x - x_mean) / (x_std)
        return x


#class Pad(nn.Module):
#    def __init__(self, size = 0):
#        super(Pad, self).__init__()
#        self.size = size
#
#    def forward(self, x):
#        if self.size > 0:
#            x = F.pad(x,[self.size]*6,mode='replicate')
#        return x

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
'''
class conv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups):
        super(conv, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False, groups=groups)

    def forward(self, x):
        #out = x
        out = self.conv1(x)
        return out

'''

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
        self.relu1 = nn.ReLU(inplace=True)#nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)#nn.ReLU(inplace=True)
        self.norm1 = nn.InstanceNorm3d(num_features=out_channels,affine=True)
        self.norm2 = nn.InstanceNorm3d(num_features=out_channels,affine=True)
        #self.se = SEBlock(channel=in_channels, reduction=4)

    def forward(self, x):

        if self.downsample is not None:
            x = self.downsample(x)

        out = x
        out = self.norm1(out)
        out = self.relu1(out)
        #out = self.pad1(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.relu2(out)
        #out = self.pad2(out)
        out = self.conv2(out)

        out = x + out

        return out


class BacisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample = None):
        super(BacisBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.downsample = downsample

        self.conv1 = conv(in_channels=in_channels, out_channels=out_channels,stride=stride)
        self.conv2 = conv(in_channels=out_channels, out_channels=out_channels, stride=1)
        self.relu1 = nn.ReLU(inplace=True)#nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)#nn.ReLU(inplace=True)\

        self.norm1 = nn.InstanceNorm3d(num_features=out_channels,affine=True)#nn.GroupNorm(num_channels=in_channels,num_groups=norm_groups)#nn.InstanceNorm3d(num_features=out_channels)
        self.norm2 = nn.InstanceNorm3d(num_features=out_channels,affine=True)#nn.GroupNorm(num_channels=out_channels,num_groups=norm_groups)#nn.InstanceNorm3d(num_features=out_channels)

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
    def __init__(self, in_channels, out_channels, stride, downsample = None, width=2, compression=4):
        super(ResNextBottleneck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.downsample = downsample

        conv_groups = 4#out_channels // (width*compression)

        self.conv_pre = nn.Conv3d(in_channels=in_channels, out_channels=out_channels//compression, kernel_size=1, stride=1,
                               padding=0, bias=False, groups=1)
        self.conv1 = conv(in_channels=out_channels//compression, out_channels=out_channels//compression,stride=stride, groups=conv_groups)
        #nn.Conv3d(in_channels=out_channels//compression, out_channels=out_channels//compression, kernel_size=3, stride=stride, padding=1, bias=False, groups=conv_groups)
        #self.pad1 = Pad(size=1)
        #self.conv2 = conv(in_channels=out_channels//compression, out_channels=out_channels//compression,stride=1)
        #nn.Conv3d(in_channels=out_channels//compression, out_channels=out_channels//compression, kernel_size=3, padding=1, bias=False, groups=conv_groups)
        #self.pad2 = Pad(size=1)
        self.relu1 = nn.ReLU(inplace=True)#nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)#nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)#nn.ReLU(inplace=True)
        self.norm1 = nn.InstanceNorm3d(num_features=out_channels//compression)#nn.GroupNorm(num_channels=in_channels,num_groups=norm_groups)#nn.InstanceNorm3d(num_features=out_channels)
        self.norm2 = nn.InstanceNorm3d(num_features=out_channels//compression)#nn.GroupNorm(num_channels=out_channels,num_groups=norm_groups)#nn.InstanceNorm3d(num_features=out_channels)
        self.norm3 = nn.InstanceNorm3d(num_features=out_channels)#nn.GroupNorm(num_channels=out_channels,num_groups=norm_groups)#nn.InstanceNorm3d(num_features=out_channels)
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

        out = x + self.se(out)
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
        #self.pad_enc = Pad(size=1)
        self.conv_dec = nn.Conv3d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, padding=0,
                                  bias=False)
        #self.pad_dec = Pad(size=1)
        self.conv_output = nn.Conv3d(in_channels=input_channels, out_channels=attention_gates, kernel_size=1, padding=0,
                                  bias=False)
        #self.pad_out = Pad(size=1)

        self.relu = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()
        #self.norm1 = nn.GroupNorm(num_channels=input_channels,num_groups=4)
        #self.norm2 = nn.GroupNorm(num_channels=input_channels,num_groups=4)

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
    def __init__(self, depth, encoder_layers, number_of_channels, number_of_outputs, block=ResNextBottleneck):
        super(UNet, self).__init__()
        print('UNet {}'.format(number_of_channels))

        self.encoder_layers = encoder_layers

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
                                    bias=False, groups=4)
        self.norm_input = nn.InstanceNorm3d(num_features=self.number_of_channels[0],affine=True)

        conv_first_list = []
        for i in range(self.encoder_layers[0]):
            conv_first_list.append(self.block(in_channels=self.number_of_channels[0],out_channels=self.number_of_channels[0],stride=1))

        self.conv_first = nn.Sequential(*conv_first_list)

        self.conv_middle = nn.Conv3d(in_channels=self.number_of_channels[-1], out_channels=self.number_of_channels[-1],
                                    kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)#nn.ReLU(inplace=False)#nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

        #self.conv_output = nn.Sequential(
        #    nn.Conv3d(in_channels=self.number_of_channels, out_channels=self.number_of_channels, kernel_size=3,
        #              stride=1, padding=1, bias=False),
        #    nn.ReLU(inplace=True)
        #)

        #conv_final_list = []

        ##for i in range(self.encoder_layers[0]):
        ##    conv_final_list.append(Residual_bottleneck(in_channels=self.number_of_channels,out_channels=self.number_of_channels,stride=1))

        #for i in range(2):
        #    conv_final_list.append(nn.Conv3d(in_channels=self.number_of_channels[0], out_channels=self.number_of_channels[0],
        #                            kernel_size=3, stride=1, padding=1, bias=False, groups = 1))
        #    conv_final_list.append(nn.InstanceNorm3d(num_features=self.number_of_channels))
        #    conv_final_list.append(nn.LeakyReLU(2e-2, inplace=True))

        #self.conv_final = nn.Sequential(*conv_final_list)

        #self.conv_final_ds = nn.Sequential(
        #    nn.Conv3d(in_channels=self.number_of_channels[2], out_channels=self.number_of_channels[0],
        #                            kernel_size=3, stride=1, padding=1, bias=True, groups = 1),
        #    nn.LeakyReLU(2e-2, inplace=True),
        #    nn.Conv3d(in_channels=self.number_of_channels[0], out_channels=self.number_of_outputs-1,
        #              kernel_size=1, stride=1, padding=0, bias=True, groups=1)
        #)


        self.conv_output = nn.Conv3d(in_channels=self.number_of_channels[0],out_channels=self.number_of_outputs,kernel_size=3, stride=1,padding=1,bias=True,groups=1)

        #self.mean = Mean()
        #self.conv_output_distance = nn.Conv3d(in_channels=self.number_of_channels,out_channels=1,kernel_size=1, stride=1,padding=0,bias=True,groups=1)
        #self.conv_output_reg = nn.Conv3d(in_channels=self.number_of_channels//4,out_channels=1,kernel_size=3, stride=1,padding=0,bias=True)
        #self.conv_output_attention = nn.Conv3d(in_channels=self.number_of_channels,out_channels=1,kernel_size=3, stride=1,padding=1,bias=True)
        self.softmax = nn.Softmax(dim=1)
        #self.softmax_ds = nn.Softmax(dim=1)
        self.construct_dencoder_convs(depth=depth,number_of_channels=number_of_channels)
        self.construct_encoder_convs(depth=depth,number_of_channels=number_of_channels)
        #self.construct_pooling_convs(depth=depth,number_of_channels=number_of_channels)
        self.construct_upsampling_convs(depth=depth,number_of_channels=number_of_channels)

        #self.conv_eso = nn.Conv1d(in_channels=8,#self.number_of_channels*int(math.pow(2,depth)),
        #                          out_channels=8,#self.number_of_channels*int(math.pow(2,depth)),
        #                          kernel_size=3, stride=1, padding=1, bias=False)
        
        #self.conv_eso_deep_supervised = nn.Conv1d(in_channels=4,
        #                          out_channels=1,
        #                          kernel_size=3, stride=1, padding=1, bias=True)
        
        
        #self.conv_deep_supervised = nn.Conv3d(in_channels=self.number_of_channels*int(math.pow(2,depth-2)),
        #                          out_channels=self.number_of_outputs,
        #                          kernel_size=3, stride=1, padding=1, bias=True)
        
        #self.conv_eso_final = nn.Conv1d(in_channels=1,
        #                          out_channels=1,
        #                          kernel_size=9, stride=1, padding=4, bias=True)

        #self.norm = nn.InstanceNorm3d(num_features=8)
        #self.adaptive_pool = nn.AdaptiveMaxPool3d(output_size=(None,1,1))

        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #        m.weight.data.normal_(0, math.sqrt(2. / n))
        #        if m.bias is not None:
        #            m.bias.data.zero_()

    def _make_encoder_layer(self, in_channels, channels, blocks, stride=1, block=ResNextBottleneck):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                #DownsamplingPixelShuffle(input_channels=in_channels, output_channels=channels,ratio=stride),
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
            #nn.Conv3d(in_channels=channels,out_channels=channels,kernel_size=3,padding=1,bias=True)#Residual(channels)
            self.encoder_convs.append(conv)

    def construct_dencoder_convs(self, depth, number_of_channels):
        for i in range(depth):

            block = self.block


            conv_list = []
            for j in range(self.encoder_layers[i]):
                conv_list.append(block(in_channels=number_of_channels[i], out_channels=number_of_channels[i], stride=1))

            conv = nn.Sequential(
                *conv_list
            )

            conv1x1 = nn.Conv3d(in_channels=2*number_of_channels[i], out_channels=number_of_channels[i], kernel_size=1, padding=0, bias=False)
            self.decoder_convs.append(conv)
            self.decoder_convs1x1.append(conv1x1)

    def construct_upsampling_convs(self, depth, number_of_channels):
        for i in range(depth-1):
            conv = nn.ConvTranspose3d(in_channels=number_of_channels[i+1],out_channels=number_of_channels[i], kernel_size=2,stride=2,padding=0,bias=False)

            #conv = UpsamplingPixelShuffle(input_channels=number_of_channels[i+1], output_channels=number_of_channels[i])

            #self_attention = Attention(channels_out, attention_gates=self.number_of_outputs - 1)

            self.upsampling.append(conv)
            #self.attention_convs.append(self_attention)


    def forward(self, x):
        skip_connections = []
        gc.collect()
        input = x[0]

        N, C, W, H, D = input.shape


        #in_norm = self.LCNorm(input)
        #in_conc = torch.cat((input,in_norm),dim=1)

        conv = self.conv_input(input)
        conv = self.norm_input(conv)
        conv = self.conv_first(conv)
        
        for i in range(self.depth-1):
            skip_connections.append(conv)
            conv = self.encoder_convs[i](conv)

        #conv = self.conv_middle(conv)
        #conv = self.relu(conv)

        #up_list = []

        for i in reversed(range(self.depth-1)):
            conv = self.upsampling[i](conv)
            conv = self.relu(conv)

            #conv = self.attention_convs[i](skip_connections[i], conv)

            conc = torch.cat([skip_connections[i],conv],dim=1)
            conv = self.decoder_convs1x1[i](conc)
            conv = self.relu(conv)
            conv = self.decoder_convs[i](conv)
            #up_list.append(conv)



        #out = self.conv_final(conv)
        #out_ds = self.conv_final_ds(up_list[-3])

        out_logits = self.conv_output(conv)

        out_logits = self.softmax(out_logits)
        #out_logits_ds = self.sigmoid(out_ds)

        #print('torch mean ', mean)

        return [out_logits]


class UNet_hardcoded(nn.Module):
    def __init__(self, number_of_channels, number_of_outputs):
        super(UNet_hardcoded, self).__init__()
        print('UNet {}'.format(number_of_channels))

        self.number_of_channels = number_of_channels
        self.number_of_outputs = number_of_outputs

        self.conv_input = nn.Conv3d(in_channels=1, out_channels=self.number_of_channels, kernel_size=3, stride=1,
                                    padding=1,
                                    bias=False)

        self.encoder_conv0_1 = Residual(in_channels=self.number_of_channels,out_channels=self.number_of_channels,stride=1)
        self.encoder_conv0_2 = Residual(in_channels=self.number_of_channels,out_channels=self.number_of_channels,stride=1)

        self.encoder_conv1_1 = Residual(in_channels=2*self.number_of_channels, out_channels=2*self.number_of_channels,
                                        stride=1)
        self.encoder_conv1_2 = Residual(in_channels=2*self.number_of_channels, out_channels=2*self.number_of_channels,
                                        stride=1)

        self.encoder_conv2_1 = Residual(in_channels=4 * self.number_of_channels,
                                        out_channels=4 * self.number_of_channels,
                                        stride=1)
        self.encoder_conv2_2 = Residual(in_channels=4 * self.number_of_channels,
                                        out_channels=4 * self.number_of_channels,
                                        stride=1)

        self.encoder_conv3_1 = Residual(in_channels=8 * self.number_of_channels,
                                        out_channels=8 * self.number_of_channels,
                                        stride=1)
        self.encoder_conv3_2 = Residual(in_channels=8 * self.number_of_channels,
                                        out_channels=8 * self.number_of_channels,
                                        stride=1)

        self.pooling_conv1 = Residual(in_channels=self.number_of_channels,out_channels=2*self.number_of_channels,stride=2,downsample=nn.Sequential(
                nn.Conv3d(in_channels=self.number_of_channels, out_channels=2*self.number_of_channels, kernel_size=2, stride=2, bias=False),
                nn.GroupNorm(num_channels=2*self.number_of_channels, num_groups=4)#nn.InstanceNorm3d(channels),
                #nn.Dropout3d(p = 0.5, inplace=True)
                ))

        self.pooling_conv2 = Residual(in_channels=self.number_of_channels, out_channels=4 * self.number_of_channels,
                                      stride=4, downsample=nn.Sequential(
                nn.Conv3d(in_channels=self.number_of_channels, out_channels=4 * self.number_of_channels, kernel_size=4,
                          stride=4, bias=False),
                nn.GroupNorm(num_channels=4 * self.number_of_channels, num_groups=4)  # nn.InstanceNorm3d(channels),
                # nn.Dropout3d(p = 0.5, inplace=True)
            ))

        self.pooling_conv3 = Residual(in_channels=self.number_of_channels, out_channels=8 * self.number_of_channels,
                                      stride=8, downsample=nn.Sequential(
                nn.Conv3d(in_channels=self.number_of_channels, out_channels=8 * self.number_of_channels, kernel_size=8,
                          stride=8, bias=False),
                nn.GroupNorm(num_channels=8 * self.number_of_channels, num_groups=4)  # nn.InstanceNorm3d(channels),
                # nn.Dropout3d(p = 0.5, inplace=True)
            ))


        self.upsampling3 = UpsamplingPixelShuffle(input_channels=8*self.number_of_channels, output_channels=self.number_of_channels,ratio=8)
        self.upsampling2 = UpsamplingPixelShuffle(input_channels=4 * self.number_of_channels,
                                                  output_channels=self.number_of_channels,ratio=4)
        self.upsampling1 = UpsamplingPixelShuffle(input_channels=2 * self.number_of_channels,
                                                  output_channels=self.number_of_channels,ratio=2)


        self.relu = nn.ReLU(inplace=False)#nn.LeakyReLU(0.2, inplace=True)
        self.conv_output = nn.Conv3d(in_channels=4*self.number_of_channels,out_channels=self.number_of_outputs-1,kernel_size=3, stride=1,padding=1,bias=True)
        self.softmax = nn.Sigmoid()

        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #        m.weight.data.normal_(0, math.sqrt(2. / n))
        #        if m.bias is not None:
        #            m.bias.data.zero_()


    def forward(self, x):
        skip_connections = []
        gc.collect()
        input = x[0]

        ci = self.conv_input(input)
        lv0 = self.encoder_conv0_1(ci)
        lv0 = self.encoder_conv0_2(lv0)

        pc1 = self.pooling_conv1(ci)
        lv1 = self.encoder_conv1_1(pc1)
        lv1 = self.encoder_conv1_2(lv1)

        pc2 = self.pooling_conv2(ci)
        lv2 = self.encoder_conv2_1(pc2)
        lv2 = self.encoder_conv2_2(lv2)

        pc3 = self.pooling_conv3(ci)
        lv3 = self.encoder_conv3_1(pc3)
        lv3 = self.encoder_conv3_2(lv3)

        up1 = self.upsampling1(lv1)
        up2 = self.upsampling2(lv2)
        up3 = self.upsampling3(lv3)


        cat1 = torch.cat([lv0,up1,up2,up3], dim=1)

        out = self.conv_output(cat1)
        out = self.softmax(out)

        return [out]

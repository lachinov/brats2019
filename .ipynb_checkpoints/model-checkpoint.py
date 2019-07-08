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

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample = None, conv_groups=1, norm_groups=4):
        super(Residual, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.downsample = downsample

        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False, groups=conv_groups)
        #self.pad1 = Pad(size=1)
        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False, groups=conv_groups)
        #self.pad2 = Pad(size=1)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)#nn.ReLU(inplace=True)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)#nn.ReLU(inplace=True)
        self.norm1 = nn.GroupNorm(num_channels=in_channels,num_groups=norm_groups)#nn.InstanceNorm3d(num_features=out_channels)
        self.norm2 = nn.GroupNorm(num_channels=out_channels,num_groups=norm_groups)#nn.InstanceNorm3d(num_features=out_channels)
        #self.se = SEBlock(channel=in_channels, reduction=4)

    def forward(self, x):

        out = x
        out = self.norm1(out)
        out = self.relu1(out)
        #out = self.pad1(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.relu2(out)
        #out = self.pad2(out)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out = x + out

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

class UpsamplingPixelShuffle(nn.Module):
    def __init__(self, input_channels, output_channels,ratio=2):
        super(UpsamplingPixelShuffle, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv = nn.Conv3d(in_channels = input_channels, out_channels=output_channels*int(ratio**3), kernel_size=1, padding=0, bias=True)
        #self.pad = Pad(size=1)
        self.relu = nn.ReLU()
        self.shuffle = shuffle(ratio=ratio)

    def forward(self, x):
        #out = self.pad(x)
        out = self.conv(x)
        out = self.relu(out)
        out = self.shuffle(out)
        return out

class Attention(nn.Module):
    def __init__(self, input_channels, attention_gates=1):
        super(Attention, self).__init__()
        self.input_channels = input_channels
        self.attention_gates = attention_gates
        self.conv_enc = nn.Conv3d(in_channels = input_channels, out_channels=input_channels, kernel_size=3, padding=1, bias=False)
        #self.pad_enc = Pad(size=1)
        self.conv_dec = nn.Conv3d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, padding=1,
                                  bias=False)
        #self.pad_dec = Pad(size=1)
        self.conv_output = nn.Conv3d(in_channels=input_channels, out_channels=attention_gates, kernel_size=3, padding=1,
                                  bias=False)
        #self.pad_out = Pad(size=1)

        self.relu = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()
        self.norm1 = nn.GroupNorm(num_channels=input_channels,num_groups=4)
        self.norm2 = nn.GroupNorm(num_channels=input_channels,num_groups=4)


    def forward(self, from_encoder, from_decoder):
        #enc = self.pad_enc(from_encoder)
        enc = self.conv_enc(from_encoder)
        enc = self.norm1(enc)
        enc = self.relu(enc)

        #dec = self.pad_dec(from_decoder)
        dec = self.conv_dec(from_decoder)
        dec = self.norm2(dec)
        dec = self.relu(dec)

        out = enc + dec
        #out = self.pad_out(out)
        out = self.conv_output(out)
        out = self.sigmoid(out)

        N, C, D, H, W = from_decoder.shape
        out = out.view(N, self.attention_gates, 1, D, H, W)
        dec = from_decoder.view(N, self.attention_gates,  C // self.attention_gates, D, H, W)

        return (dec*out).view(N,C,D,H,W)

class UNet(nn.Module):
    def __init__(self, depth, encoder_layers, number_of_channels, number_of_outputs):
        super(UNet, self).__init__()
        print('UNet {}'.format(number_of_channels))

        self.encoder_layers = encoder_layers

        self.number_of_channels = number_of_channels
        self.number_of_outputs = number_of_outputs
        self.depth = depth

        self.conv_input = 0

        self.encoder_convs = nn.ModuleList()

        #self.pooling_covs = nn.ModuleList()

        self.upsampling = nn.ModuleList()

        self.decoder_convs = nn.ModuleList()

        self.decoder_convs1x1 = nn.ModuleList()

        self.attention_convs = nn.ModuleList()

        self.upsampling_distance = nn.ModuleList()

        #self.padding1 = Pad(size=1)
        self.conv_input = nn.Conv3d(in_channels=1, out_channels=self.number_of_channels, kernel_size=3, stride=1, padding=1,
                                    bias=False)
        self.conv_middle = nn.Conv3d(in_channels=self.number_of_channels*int(math.pow(2,depth)), out_channels=self.number_of_channels*int(math.pow(2,depth)),
                                    kernel_size=3, stride=1, padding=1,
                                    bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)#nn.ReLU(inplace=False)#nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

        num_of_channels_final = self.number_of_channels

        self.conv_output = nn.Sequential(
            nn.Conv3d(in_channels=self.number_of_channels, out_channels=self.number_of_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

        #Residual(in_channels=self.number_of_channels,out_channels=self.number_of_channels,stride=1,conv_groups=4, norm_groups=8)
        #nn.Conv3d(in_channels=self.number_of_channels,out_channels=self.number_of_channels,kernel_size=3, stride=1,padding=1,bias=True)
        self.conv_output1 = nn.Sequential(
            nn.Conv3d(in_channels=self.number_of_channels, out_channels=self.number_of_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        #Residual(in_channels=self.number_of_channels,out_channels=num_of_channels_final,stride=1,conv_groups=4, norm_groups=8)
        #nn.Conv3d(in_channels=self.number_of_channels,out_channels=self.number_of_channels,kernel_size=3, stride=1,padding=1,bias=True)
        #self.conv_output2 = Residual(in_channels=self.number_of_channels,out_channels=self.number_of_channels,stride=1,norm_groups=8)
        #nn.Sequential(
        #    Pad(size=1),
        #    nn.Conv3d(in_channels=self.number_of_channels, out_channels=self.number_of_channels, kernel_size=3,
        #              stride=1, padding=0, bias=False),
        #    nn.ReLU(inplace=True)
        #)
        #Residual(in_channels=num_of_channels_final,out_channels=num_of_channels_final,stride=1,conv_groups=4, norm_groups=8)
        #nn.Conv3d(in_channels=self.number_of_channels,out_channels=self.number_of_channels,kernel_size=3, stride=1,padding=1,bias=True)
        self.conv_output3 = nn.Sequential(
            #Pad(size=1),
            nn.Conv3d(in_channels=self.number_of_channels, out_channels=self.number_of_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        #Residual(in_channels=num_of_channels_final,out_channels=num_of_channels_final,stride=1,conv_groups=4, norm_groups=8)
        #nn.Conv3d(in_channels=self.number_of_channels,out_channels=self.number_of_channels,kernel_size=3, stride=1,padding=1,bias=True)
        self.conv_output4 = nn.Conv3d(in_channels=num_of_channels_final,out_channels=self.number_of_outputs-1,kernel_size=3, stride=1,padding=1,bias=True,groups=4)
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
        
        
        self.conv_deep_supervised = nn.Conv3d(in_channels=self.number_of_channels*int(math.pow(2,depth-2)),
                                  out_channels=self.number_of_outputs-1,
                                  kernel_size=3, stride=1, padding=1, bias=True)
        
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

    def _make_encoder_layer(self, in_channels, channels, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv3d(in_channels=in_channels, out_channels=channels, kernel_size = 1, stride= stride, bias=False),
                nn.GroupNorm(num_channels=channels, num_groups=4)#nn.InstanceNorm3d(channels),
                #nn.Dropout3d(p = 0.5, inplace=True)
                )
            #downsample = nn.Sequential(
            #   nn.MaxPool3d(kernel_size=2,stride=2),
            #    nn.Conv3d(in_channels=in_channels, out_channels=channels, kernel_size=1, stride=1, bias=False)
            #)

        layers = []
        layers.append(Residual(in_channels=in_channels, out_channels=channels, stride=stride, downsample=downsample))
        for _ in range(1, blocks):
            layers.append(Residual(in_channels=channels,out_channels=channels,stride=1))

        return nn.Sequential(*layers)

    def construct_encoder_convs(self, depth, number_of_channels):
        for i in range(depth):
            channels = number_of_channels*int(math.pow(2,i))
            conv = self._make_encoder_layer(in_channels=channels,channels=channels*2,blocks=self.encoder_layers[i],stride=2)
            #nn.Conv3d(in_channels=channels,out_channels=channels,kernel_size=3,padding=1,bias=True)#Residual(channels)
            self.encoder_convs.append(conv)

    def construct_dencoder_convs(self, depth, number_of_channels):
        for i in range(depth):
            channels = number_of_channels*int(math.pow(2,i))
            conv = nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, bias=False)
            #nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, bias=True)
            #Residual(channels)
            conv1x1 = nn.Conv3d(in_channels=2*channels, out_channels=channels, kernel_size=1, bias=False)
            self.decoder_convs.append(conv)
            self.decoder_convs1x1.append(conv1x1)

    #def construct_pooling_convs(self, depth, number_of_channels):
    #    for i in range(depth):
    #       channels_in = number_of_channels * int(math.pow(2, i))
    #        channels_out = number_of_channels * int(math.pow(2, i+1))
    #        conv = nn.Conv3d(in_channels=channels_in, out_channels=channels_out, kernel_size=2, stride=2, padding=0, bias=False)
    #        self.pooling_covs.append(conv)

    def construct_upsampling_convs(self, depth, number_of_channels):
        for i in range(depth):
            channels_in = number_of_channels * int(math.pow(2, i+1))
            channels_out = number_of_channels * int(math.pow(2, i))
            #conv = nn.ConvTranspose3d(in_channels=channels_in,out_channels=channels_out, kernel_size=2,stride=2,padding=0,bias=False)

            #conv = nn.Sequential(
            #    nn.Conv3d(in_channels=channels_in, out_channels=channels_out, kernel_size=1, stride=1, padding=0,
            #              bias=False),
            #    nn.Upsample(scale_factor=2, mode='trilinear'),
            #)

            #conv_eso = nn.ConvTranspose1d(in_channels=int(math.pow(2, i+1)), out_channels=int(math.pow(2, i)),
            #                              kernel_size=2,stride=2,padding=0,bias=False)
            conv = UpsamplingPixelShuffle(input_channels=channels_in, output_channels=channels_out)
            #nn.Conv3d(in_channels=channels_in, out_channels=channels_out, kernel_size=2, stride=2, padding=0, bias=False)

            conv_distance = nn.Sequential(
                nn.ConvTranspose3d(in_channels=channels_in, out_channels=channels_out, kernel_size=2, stride=2,
                                   padding=0, bias=False, groups=4),
                nn.LeakyReLU(0.2),
                #Pad(size=1),
                nn.Conv3d(in_channels=channels_out, out_channels=channels_out, kernel_size=3, stride=1, padding=1,
                          bias=False,groups=4),
                nn.LeakyReLU(0.2),
            )

            #self_attention = Attention(channels_out, attention_gates=self.number_of_outputs-1)

            self.upsampling.append(conv)
            #self.attention_convs.append(self_attention)
            #self.upsampling_distance.append(conv_distance)


    def forward(self, x):
        skip_connections = []
        gc.collect()
        input = x[0]

        N, C, W, H, D = input.shape


        #in_norm = self.LCNorm(input)
        #in_conc = torch.cat((input,in_norm),dim=1)

        #conv = self.padding1(input)
        conv = self.conv_input(input)
        
        for i in range(self.depth):
            skip_connections.append(conv)
            conv = self.encoder_convs[i](conv)
            #conv = F.dropout(conv,p=0.7,training=self.training)
            #conv = self.relu(conv)
            #conv = self.pooling_covs[i](conv)
            #conv = self.relu(conv)

        #conv = self.padding1(conv)
        conv = self.conv_middle(conv)
        conv = self.relu(conv)
        #conv_dist = conv

        #conv_eso = self.norm(conv[:,:8,:,:,:])
        #n,c,h,w, d = conv_eso.shape
        #conv_eso = self.adaptive_pool(conv_eso).view(n,c,-1)#torch.max(conv_eso,dim=(2,3))
        #conv_eso = self.conv_eso(conv_eso)
        #conv_eso = self.relu(conv_eso)


        conv_up = []
        #conv_eso_up = []
        for i in reversed(range(self.depth)):
            #conv_dist = self.upsampling_distance[i](conv_dist)
            conv = self.upsampling[i](conv)
            conv = self.relu(conv)

            #conv = self.attention_convs[i](skip_connections[i], conv)

            conc = torch.cat([skip_connections[i],conv],dim=1)
            conv = self.decoder_convs1x1[i](conc)
            conv = self.relu(conv)
            #conv = self.padding1(conv)
            conv = self.decoder_convs[i](conv)
            conv = self.relu(conv)
            
            conv_up.append(conv)

            #conv_eso = self.upsampling_eso[i](conv_eso)
            #conv_eso = self.relu(conv_eso)

            #conv_eso_up.append(conv_eso)


        #eso_deep_supervised = self.conv_eso_deep_supervised(conv_eso_up[0])
        #eso_deep_supervised = self.sigmoid(eso_deep_supervised).view(N, 1, 1, 1, -1)

        #ds = self.padding1(conv_up[1])
        out_ds = self.conv_deep_supervised(conv_up[1])

        #bg_ds = out_ds[:,:1]
        #fg_ds = out_ds[:,1:] * (eso_deep_supervised.detach() > 0.5).float()
        #out_ds = torch.cat((bg_ds,fg_ds),dim=1)
        #out_ds = self.sigmoid(out_ds)
            
        #conv_eso = self.conv_eso_final(conv_eso)
        #conv_eso = self.sigmoid(conv_eso).view(N, 1, 1, 1, D)

        conv123 = self.conv_output(conv)
        #out = self.relu(out)
        conv = self.conv_output1(conv123)
        #conv = self.relu(conv)
        #conv = self.conv_output2(conv)
        #conv = self.relu(conv)
        conv = self.conv_output3(conv)
        #conv = self.relu(conv)
        #conv = self.padding1(conv)
        out = self.conv_output4(conv)

        #conv_dist = self.padding1(conv123[:,:self.number_of_channels//4])
        #out_reg = self.conv_output_reg(conv_dist)
        #tensors = list(torch.chunk(out,5,dim=1))
        #tensors[1] = tensors[1]*conv_eso.detach()
        #out = torch.cat(tensors,dim=1)
        #
        #fg = out[:,1:] * (conv_eso.detach() > 0.5).float()
        #out = out*(conv_eso.detach() > 0.5).float()
        #print(bg.shape, fg.shape)
        #out = torch.cat((bg,fg),dim=1)
        #out_att = self.conv_output_attention(conv)

        #out_att = self.adaptive_pool(out_att).view(N, 1, -1)
        #out = self.softmax(out)
        #out_att = self.sigmoid(out_att).view(N,1,-1,1,1)
        #out = out ** 2

        #out = self.softmax(out)
        out = self.sigmoid(out)
        out_ds = self.sigmoid(out_ds)

        #out = out * (out_att.detach() > 0.5).float()

        #multiplier = (out_att.detach() > 0.5).float()

        #if self.training:
        #multiplier = x[1]

        #bg = out[:, :1]
        #fg = out[:, 1:] * multiplier
        #out = torch.cat((bg, fg), dim=1)

        #if self.training:
        return [out, out_ds]#, ]
        
        #return [out,out_att]#, out_ds, conv_eso, eso_deep_supervised]#[out, out_ds]


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


        self.relu = nn.LeakyReLU(0.2, inplace=False)#nn.LeakyReLU(0.2, inplace=True)
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

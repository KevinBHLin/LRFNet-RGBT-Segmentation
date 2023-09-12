import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
#regular
from torchvision.models import resnet18 as Resnet18
import torchvision.models as models
from torch.autograd import Variable
from torch.distributions import Normal, Independent, kl
from torch.nn import BatchNorm2d
class channel_fusion(nn.Module):
    def __init__(self,input_channels,out_channels):
        super(channel_fusion, self).__init__()
        self.reduct_channel = out_channels//2
        self.conv1 = nn.Sequential(nn.Conv2d(input_channels,out_channels,kernel_size=3,dilation=1,padding=1,bias=False),
                                    nn.BatchNorm2d(out_channels),
                                    nn.LeakyReLU(0.2),
                                    )
        self.fc = nn.Sequential(nn.Linear(out_channels,self.reduct_channel,bias=False),
                                nn.LeakyReLU(0.2),
                                nn.Linear(self.reduct_channel,out_channels,bias=False),
                                #nn.Sigmoid()
                                    )

        self.sigmoid = nn.Sigmoid()
    def forward(self,x):  

        x = self.conv1(x)
        origin = x
        x1 = x.mean(dim=(2,3),keepdim=False)
        x2 = (x.max(dim=3,keepdim=False)[0]).max(dim=2,keepdim=False)[0]
        x1 = self.fc(x1)
        x2 = self.fc(x2)
        x = self.sigmoid(x1+x2)
        x = origin*x.unsqueeze(2).unsqueeze(3)
        #x = self.conv2(x)
        return x
class LFM(nn.Module):
    def __init__(self,input_channels,reduction):
        super(LFM, self).__init__()
        self.reduct_channel = input_channels//reduction
        self.latent_dim = 8
        self.fc_Down = channel_fusion(input_channels,self.reduct_channel)
        self.mu = nn.Sequential(
                                nn.Conv2d(self.reduct_channel,self.latent_dim,kernel_size=1,dilation=1,padding=0,bias=False),   
                                    )
        self.fc_Up = nn.Sequential(                                                                                       
                                    nn.Conv2d(self.latent_dim,1,kernel_size=1,dilation=1,padding=0,bias=False),
                                    #nn.BatchNorm2d(1),
                                    nn.Sigmoid()
                                    )       
        self.upsample8 = nn.Upsample(scale_factor=8,mode='bilinear',align_corners=False)
    def forward(self,rgb,inf):

        z = torch.cat([rgb,inf],dim=1)
        z = self.fc_Down(z)
        mu = self.mu(z)
        w = self.fc_Up(mu)
        
        return w*rgb+(1-w)*inf,w

 
class Encoder(nn.Module):
    def __init__(self,num_resnet_layers):
        super(Encoder, self).__init__()
        self.num_resnet_layers = num_resnet_layers
        resnet_raw_model = eval('models.resnet'+str(self.num_resnet_layers))(pretrained=True)#读取预训练模型
        self.encoder_conv1 = resnet_raw_model.conv1              
        self.encoder_bn1 = resnet_raw_model.bn1
        self.encoder_relu = resnet_raw_model.relu
        self.encoder_maxpool = resnet_raw_model.maxpool
        self.encoder_layer1 = resnet_raw_model.layer1
        self.encoder_layer2 = resnet_raw_model.layer2
        self.encoder_layer3 = resnet_raw_model.layer3
        self.encoder_layer4 = resnet_raw_model.layer4  
        ########################################
  
    def forward(self, input):
        # encoder
        context_stage = {}
        input = self.encoder_conv1(input)
        input = self.encoder_bn1(input)
        input = self.encoder_relu(input)
        #unet_stage[0] = input
        input = self.encoder_maxpool(input)       
        ######################################################################
        input = self.encoder_layer1(input)
        #unet_stage[1] = input
        ######################################################################
        input = self.encoder_layer2(input)
        #unet_stage[2] = input
        ######################################################################
        input = self.encoder_layer3(input)
        context_stage[3] = input
        ######################################################################
        input = self.encoder_layer4(input)
        context_stage[4] = input

        return context_stage[3],context_stage[4]  
class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class UpSample(nn.Module):

    def __init__(self, n_chan, factor=2):
        super(UpSample, self).__init__()
        out_chan = n_chan * factor * factor
        self.proj = nn.Conv2d(n_chan, out_chan, 1, 1, 0)
        self.up = nn.PixelShuffle(factor)
        self.init_weight()

    def forward(self, x):
        feat = self.proj(x)
        feat = self.up(feat)
        return feat

    def init_weight(self):
        nn.init.xavier_normal_(self.proj.weight, gain=1.)


class BiSeNetOutput(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes, up_factor=32, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.up_factor = up_factor
        out_chan = n_classes
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, out_chan, kernel_size=1, bias=True)
        self.up = nn.Upsample(scale_factor=up_factor,
                mode='bilinear', align_corners=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        x = self.up(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size= 1, bias=False)
        self.bn_atten = BatchNorm2d(out_chan)
        #  self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        #  atten = self.sigmoid_atten(atten)
        atten = atten.sigmoid()
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class ContextRefine(nn.Module):
    def __init__(self):
        super(ContextRefine, self).__init__()

        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)
        self.up32 = nn.Upsample(scale_factor=2.)
        self.up16 = nn.Upsample(scale_factor=2.)

        self.init_weight()
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
    def forward(self, feat16, feat32):

        avg = torch.mean(feat32, dim=(2, 3), keepdim=True)
        avg = self.conv_avg(avg)

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg
        feat32_up = self.up32(feat32_sum)
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = self.up16(feat16_sum)
        feat16_up = self.conv_head16(feat16_up)

        return feat16_up # x8, x16
class ContextPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ContextPath, self).__init__()
        self.resnet = Resnet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)
        self.up32 = nn.Upsample(scale_factor=2.)
        self.up16 = nn.Upsample(scale_factor=2.)

        self.init_weight()

    def forward(self, x):
        feat8, feat16, feat32 = self.resnet(x)

        avg = torch.mean(feat32, dim=(2, 3), keepdim=True)
        avg = self.conv_avg(avg)

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg
        feat32_up = self.up32(feat32_sum)
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = self.up16(feat16_sum)
        feat16_up = self.conv_head16(feat16_up)

        return feat16_up, feat32_up # x8, x16

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class SpatialPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, ks=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(64, 128, ks=1, stride=1, padding=0)
        self.init_weight()

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.conv_out(feat)
        return feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        ## use conv-bn instead of 2 layer mlp, so that tensorrt 7.2.3.4 can work for fp16
        self.conv = nn.Conv2d(out_chan,
                out_chan,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.bn = nn.BatchNorm2d(out_chan)
        #  self.conv1 = nn.Conv2d(out_chan,
        #          out_chan//4,
        #          kernel_size = 1,
        #          stride = 1,
        #          padding = 0,
        #          bias = False)
        #  self.conv2 = nn.Conv2d(out_chan//4,
        #          out_chan,
        #          kernel_size = 1,
        #          stride = 1,
        #          padding = 0,
        #          bias = False)
        #  self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv(atten)
        atten = self.bn(atten)
        #  atten = self.conv1(atten)
        #  atten = self.relu(atten)
        #  atten = self.conv2(atten)
        atten = atten.sigmoid()
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class LRFNet(nn.Module):

    def __init__(self, n_class,num_resnet_layers):
        super(LRFNet, self).__init__()
        self.RGB_context_encoder = Encoder(num_resnet_layers=num_resnet_layers)  
        self.Thermal_context_encoder = Encoder(num_resnet_layers=num_resnet_layers) 
        self.RGB_spatial_encoder = SpatialPath()
        self.Thermal_spatial_encoder = SpatialPath()

        self.downsample2 = nn.Upsample(scale_factor=1/2)
        self.downsample8 = nn.Upsample(scale_factor=1/8)

        self.VFFM8 = LFM(128*2,16)

        self.VFFM_cp = LFM(128*2,16)

        self.context_refine_rgb = ContextRefine()
        self.context_refine_inf = ContextRefine()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_class, up_factor=8)
    def forward(self, input,istrain=False):
        rgb = input[:,:3]
        thermal = input[:,3:] 
        thermal = thermal.repeat(1,3,1,1)   
        w = {}
        #feature extraction
        rgb8 = self.RGB_spatial_encoder(rgb)
        inf8 = self.Thermal_spatial_encoder(thermal)
        rgb16,rgb32 = self.RGB_context_encoder(rgb)
        inf16,inf32 = self.Thermal_context_encoder(thermal)
        #context features refine
        rgb_cp8 = self.context_refine_rgb(rgb16,rgb32)
        inf_cp8 = self.context_refine_inf(inf16,inf32)
        #modality features fusion
        #labels = self.downsample8(labels) if istrain is True else 0
        feat_sp8,w[0] = self.VFFM8(rgb8,inf8)
        feat_cp8,w[1] = self.VFFM_cp(rgb_cp8,inf_cp8)

        #multi level features fusion
        feat_fuse = self.ffm(feat_sp8, feat_cp8)
        #decoder
        precdict = self.conv_out(feat_fuse)
        if istrain is False:
            return precdict
        else:
            return precdict,w

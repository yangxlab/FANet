import copy

import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from resnet import resnet50, resnet18
import torch.nn.functional as F
import math
from einops import rearrange, repeat

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class FeatureBlock(nn.Module):
    def __init__(self, input_dim, low_dim, dropout=0.5, relu=True):
        super(FeatureBlock, self).__init__()
        feat_block = []
        feat_block += [nn.Linear(input_dim, low_dim)]
        feat_block += [nn.BatchNorm1d(low_dim)]

        feat_block = nn.Sequential(*feat_block)
        feat_block.apply(weights_init_kaiming)
        self.feat_block = feat_block

    def forward(self, x):
        x = self.feat_block(x)
        return x
class AGP(nn.Module):
    def __init__(self, K, dim):
        super(AGP, self).__init__()
        self.gem = GeM()
        self.K = K
        self.Kconv = nn.ModuleList()
        for i in range(self.K):
            con1 = nn.Conv2d(dim, dim//self.K, kernel_size=1, stride=1, padding=0)
            self.Kconv.append(con1)
        self.Kconv.apply(weights_init_kaiming)
        self.SE = ChannelAttention(dim)

    def forward(self, x):
        b,c,h,w = x.size()
        x_fft = torch.fft.fft2(x)
        x_amp = torch.abs(x_fft)
        x_pha = torch.angle(x_fft)
        # print(type(x_fft),type(x_amp),type(x_pha))
        # print(x_fft,x_amp,x_pha)
        x_g = rearrange(x_amp, 'b c h w -> b c (h w)')
        x_g = self.gem(x_g,dim=-1)
        x_g = x_g.view(b,c,1,-1)
        x_a = []
        for i in range(self.K):
            x_i = self.Kconv[i](x_g)
            x_a.append(x_i)
        x_a = torch.cat(x_a, dim=1)
        x_se = self.SE(x_a)
        x_pha = x_se*x_pha + x_pha
        x_f = torch.fft.ifft2(torch.complex(x_amp, x_pha))
        # print(type(x_f))
        # print(x_f)
        return x_f.real
class ANM(nn.Module):
    def __init__(self, dim):
        super(ANM, self).__init__()
        self.res1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(dim),
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        x_fft = torch.fft.fft2(x)
        x_amp = torch.abs(x_fft)
        x_pha = torch.angle(x_fft)
        x_r1 = self.res1(x_amp)
        x_r2 = self.res2(x_amp)
        f1 = torch.fft.ifft2(torch.complex(x_r1, x_pha))
        f2 = torch.fft.ifft2(torch.complex(x_r2, x_pha))

        return f1.real,f2.real
class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out)
        return output
    
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=0.5, relu=True):
        super(ClassBlock, self).__init__()
        classifier = []
        if relu:
            classifier += [nn.LeakyReLU(0.1)]
        if dropout:
            classifier += [nn.Dropout(p=dropout)]

        classifier += [nn.Linear(input_dim, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.classifier = classifier

    def forward(self, x):
        x = self.classifier(x)
        return x

class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        x = self.visible.layer1(x)
        # x = self.visible.layer2(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        x = self.thermal.layer1(x)
        # x = self.thermal.layer2(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.AGP1 = AGP(K=8,dim=256)
        # self.AGP2 = AGP(K=8,dim=512)
        # self.ANM = ANM(dim=2048)
        self.base = model_base
        self.layer4 = copy.deepcopy(self.base.layer4)

    def forward(self, x):
        # x = self.base.layer1(x)
        # x = self.AGP1(x)
        x = self.base.layer2(x)
        # x = self.AGP2(x)
        x = self.base.layer3(x)
        t_x = self.layer4(x)
        x = self.base.layer4(x)
        # x = self.ANM(x)
        # x = self.layer4(x)
        # x = self.base.layer4(x)
        return x,t_x


class GeM1d(nn.Module):
    def __init__(self, p=3.5):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
    def forward(self, feat, dim):   
        feat = (torch.mean(feat**self.p, dim=dim) + 1e-12)**(1/self.p)
        return feat
class GeM(nn.Module):
    def __init__(self, p=3.5, tenson="1D"):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        assert tenson in ["1D", "2D", "3D"]
        self.tenson = tenson
    def forward(self, feat, dim): 
        if self.tenson == "1D":
            feat = (torch.mean(feat**self.p, dim=dim) + 1e-12)**(1/self.p)
        elif self.tenson == "2D":
            feat = F.avg_pool2d(feat.clamp(min=1e-12).pow(self.p), (feat.size(-2), feat.size(-1))).pow(1. / self.p) 
        else:
            feat = F.avg_pool3d(feat.clamp(min=1e-12).pow(self.p), (feat.size(-3),feat.size(-2), feat.size(-1))).pow(1. / self.p) 
        return feat
class MLP(nn.Module):
    def __init__(self, dimin, dim,dimout, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dimin),
            nn.Linear(dimin, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dimout),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class embed_net(nn.Module):
    def __init__(self,  class_num, drop=0.2, flag="global", arch='resnet50', num_stripes = 4):
        super(embed_net, self).__init__()
        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)
        pool_dim = 2048
        self.dropout = drop
        self.flag = flag
        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck1 = nn.BatchNorm1d(pool_dim)
        self.bottleneck1.bias.requires_grad_(False)  # no shift
        self.bottleneck1.apply(weights_init_kaiming)
        self.bottleneck2 = nn.BatchNorm1d(pool_dim)
        self.bottleneck2.bias.requires_grad_(False)  # no shift
        self.bottleneck2.apply(weights_init_kaiming)
        self.classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier1 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier1.apply(weights_init_classifier)
        self.classifier2 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier2.apply(weights_init_classifier)
        self.num_stripes=num_stripes
        if self.flag == "local":
            self.gem = GeM(tenson="1D")
            self.gem_list = nn.ModuleList()
            for i in range(0,self.num_stripes):
                gem = GeM(tenson="1D")
                self.gem_list.append(gem)
        elif self.flag == "global_local":
            self.gem = GeM(tenson="1D")
            self.gem1 = GeM(tenson="1D")
            self.gem2d = GeM(tenson="1D")
            self.gem_list = nn.ModuleList()
            for i in range(0,self.num_stripes):
                gem = GeM(tenson="1D")
                self.gem_list.append(gem)
            self.mlp = MLP(dimin = 2*pool_dim, dim = pool_dim, dimout=pool_dim)
        else:
            self.gem = GeM(tenson="1D")
            self.gem1 = GeM(tenson="1D")
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    # def pcb(self, feat,num_stripes,seq_len):
        
    #     stripe_h = int(feat.size(2) / num_stripes)
    #     local_feat_list = []
    #     logits_list = []
    #     for i in range(num_stripes):
    #         # shape [NT, C, 1, 1]
    #         # gm pool
    #         local_feat = feat[:, :, i * stripe_h: (i + 1) * stripe_h, :]
    #         bt, c, h, w = local_feat.shape 
    #         local_feat = self.gem2d_list[i](local_feat,dim=-1)
                
    #         # shape [NT, c, 1, 1]
    #         local_feat = self.local_conv_list[i](local_feat.view(feat.size(0),feat.size(1),1,1))
    #         # shape [NT, c]
    #         local_feat = local_feat.view(local_feat.size(0), -1)

    #         local_feat = rearrange(local_feat, '(b t) n->b t n', t=seq_len)
    #         local_feat = self.gem1d_list[i](local_feat,dim=1)

    #         local_feat_list.append(local_feat)
    #         logits_list.append(self.fc_list[i](local_feat))

    #         feat_all = [lf for lf in local_feat_list]
    #         feat_all = torch.cat(feat_all, dim=1)

    #     return local_feat_list, logits_list, feat_all 
    def partition(self, x, stride):
        bt, c, h, w = x.size()
        feat = []
        for i in range(0, stride):
            xi = x[:,:,i *(h // stride): (i + 1) * (h // stride),:]
            # xi = F.avg_pool2d(xi, xi.size()[2:])
            xi = rearrange(xi, 'bt c h w->bt c (h w)')
            xi = self.gem_list[i](xi,dim=-1)
            feat.append(xi)
        x = torch.cat(feat, dim=-1)
        x = x.view(bt,c,-1)
        return x
    def forward(self, x1, x2, m1, m2, modal=0, seq_len = 8):
        b, c, h, w = x1.size()
        t = seq_len
        # if self.training:
        #     x1 = x1*m1+x1
        #     x2 = x2*m2+x2
        x1 = x1.view(int(b * t), int(c / seq_len), h, w)
        x2 = x2.view(int(b * t), int(c / seq_len), h, w)
        

        # style augmentation
        # if self.training:
        #     # IR modality

        #     frame_batch = seq_len*16
        #     delta = torch.rand(frame_batch) + 0.5*torch.ones(frame_batch) # [0.5-1.5]
        #     inter_map = delta.unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1).cuda()
        #     x2 = x2*inter_map
            
        #     # RGB modality
        #     alpha = (torch.rand(frame_batch) + 0.5*torch.ones(frame_batch)).unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1)
        #     beta = (torch.rand(frame_batch) + 0.5*torch.ones(frame_batch)).unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1)
        #     gamma = (torch.rand(frame_batch) + 0.5*torch.ones(frame_batch)).unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1)
        #     inter_map = torch.cat((alpha, beta, gamma), dim=1).cuda()
        #     x1 = x1*inter_map
        #     for i in range(x1.shape[0]):
        #         x1[i] = x1[i,torch.randperm(3),:,:] 
        # SAM augmentation
        if self.training:
            # IR modality
            # 16 18 288 144           96 3 288 144
            m2 = m2.view(int(b * t), int(c / seq_len), h, w)
            m1 = m1.view(int(b * t), int(c / seq_len), h, w)
            # print(x1.shape,x2.shape,m1.shape,m2.shape)
            frame_batch = seq_len*16
            # delta = torch.rand(1) + 0.5*torch.ones(1) # [0.5-1.5]
            # delta = (torch.rand(frame_batch) + 0.1*torch.ones(frame_batch)).unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1).cuda()*m2
            delta = torch.rand(frame_batch).unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1).cuda()*m2
            # x2 = x2*delta+x2
            x2 = x2+delta
            
            # RGB modality
            # alpha = (torch.rand(frame_batch) + 0.1*torch.ones(frame_batch)).unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1).cuda()*m1
            alpha = torch.rand(frame_batch).unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1).cuda()*m1
            # alpha = (torch.rand(1) + 0.5*torch.ones(frame_batch)).cuda()*m1
            # x1 = x1*alpha+x1
            x1 = alpha+x1

            # for i in range(x1.shape[0]):
            #     x1[i] = x1[i,torch.randperm(3),:,:] 

        # SAM augmentation
        # if self.training:
        #     # IR modality
        #     m2 = m2.view(int(b * t), int(c / seq_len), h, w)
        #     m1 = m1.view(int(b * t), int(c / seq_len), h, w)
        #     frame_batch = seq_len*16
        #     delta = torch.rand(frame_batch).unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1).cuda()*m2
        #     x2 = x2+delta            
        #     # RGB modality
        #     alpha = torch.rand(frame_batch).unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1).cuda()*m1
        #     x1 = alpha+x1



        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)
    
        x,_ = self.base_resnet(x)
        
        if self.flag=="local":
            # local_feat_list, logits_list, feat_all = self.pcb(x,self.num_stripes,seq_len)
            local_feat = self.partition(x, self.num_stripes).squeeze()
            local_feat = rearrange(local_feat, '(b t) n h->b (t h) n', t=seq_len)
            # print("global_feat", global_feat.shape)
            x_local = self.gem(local_feat,dim=1)
            # print("x_global",x_global.shape)
            feat_l  = self.bottleneck1(x_local)

            if self.training:
                return  x_local, self.classifier1(feat_l)
            else:
                return self.l2norm(feat_l)
        elif self.flag=="global_local":
            # local
            local_feat = self.partition(x,self.num_stripes).squeeze()
            local_feat = rearrange(local_feat, '(b t) n h->b (t h) n', t=seq_len)
            # print("global_feat", global_feat.shape)
            x_local = self.gem(local_feat,dim=1)
            # print("x_global",x_global.shape)
            feat_l  = self.bottleneck1(x_local)
            #global
            xg = rearrange(x, 'bt c h w->bt c (h w)')
            global_feat = self.gem2d(xg,dim=-1).squeeze()
            # print("global_feat", global_feat.shape)
            global_feat = rearrange(global_feat, '(b t) n->b t n', t=seq_len)
            # print("global_feat", global_feat.shape)
            x_global = self.gem1(global_feat,dim=1)
            # print("x_global",x_global.shape)
            feat_g  = self.bottleneck(x_global)

            x_feat = x_local + x_global 
            feat = self.bottleneck2(x_feat)
            if self.training:
                return x_global, self.classifier(feat_g), x_local, self.classifier1(feat_l), x_feat, self.classifier2(feat)
            else:
                return self.l2norm(x_feat)
        else:
            # global_feat = self.avgpool(x).squeeze()
            xg = rearrange(x, 'bt c h w->bt c (h w)')
            global_feat = self.gem(xg, dim=-1).squeeze()
            # print("global_feat", global_feat.shape)
            global_feat = rearrange(global_feat, '(b t) n->b t n', t=seq_len)
            # print("global_feat", global_feat.shape)
            x_global = self.gem1(global_feat,dim=1)
            # print("x_global",x_global.shape)
            feat  = self.bottleneck(x_global)
            if self.training:
                return x_global, self.classifier(feat)
            else:
                return self.l2norm(feat)
        






class show():

    def partition(self, x, stride):
        bt, c, h, w = x.size()
        feat = []
        for i in range(0, stride):
            xi = x[:,:,i *(h // stride): (i + 1) * (h // stride),:]
            xi = rearrange(xi, 'bt c h w->bt c (h w)')
            xi = self.swp_list[i](xi,dim=-1)
            feat.append(xi)
        x = torch.cat(feat, dim=-1)
        x = x.view(bt,c,-1)
        return x 

    def STAM(self,x):
        #spatial branch
        # local
        local_feat = self.partition(x,self.num_stripes).squeeze()
        local_feat = rearrange(local_feat, '(b t) n h->b (t h) n', t=seq_len)
        x_local = self.twp(local_feat,dim=1)
        feat_l  = self.bottleneck1(x_local)

        #global
        xg = rearrange(x, 'bt c h w->bt c (h w)')
        global_feat = self.swp(xg,dim=-1).squeeze()
        global_feat = rearrange(global_feat, '(b t) n->b t n', t=seq_len)
        x_global = self.twp(global_feat,dim=1)
        feat_g  = self.bottleneck2(x_global)
        x_s = torch.cat((x_local, x_global),dim=1)

        #temporal branch
        tfeat = self.GAP(x)
        t_feat = rearrange(t_feat, '(b t) n->b t n', t=seq_len)
        x_t = self.twp(t_feat,dim=1)
        feat_t  = self.bottleneck2(x_t)

        # ST branch
        xst = rearrange(x, 'bt c h w->b c t h w')
        x_st = self.stwp(xst,dim=-1).squeeze()
        feat_st  = self.bottleneck3(x_st)

        x_feat = w1*x_s + w2*x_t +w3*x_st
        feat = self.bottleneck4(x_feat) 
        if self.training:
            return x_s, self.classifier(feat_s), x_t, self.classifier1(feat_t),\
                    x_st, self.classifier1(feat_st), x_feat, self.classifier2(feat)
        else:
            return self.l2norm(x_feat)




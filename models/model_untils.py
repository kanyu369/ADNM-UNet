import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch import Tensor
from typing import Optional
import torch.utils.checkpoint as checkpoint

from einops import rearrange
import numpy as np
from timm.models.vision_transformer import _cfg
from timm.layers import trunc_normal_

from timm.layers import DropPath, to_2tuple, trunc_normal_, AvgPool2dSame, Mlp, GlobalResponseNormMlp, LayerNorm2d, LayerNorm, create_conv2d, get_act_layer, make_divisible, to_ntuple
from timm.models import register_model
from timm.models.vision_transformer import _load_weights
from models.WTConv2d import WTConv2d
import numbers
import math

def to_bchw(x):
    b, l, d = x.shape
    h = w = int(math.sqrt(l))
    return x.reshape(b,h,w,d).permute(0,3,1,2)

def to_bld(x):
    return x.flatten(2).transpose(1,2)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

        

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        out = (x-mu) / torch.sqrt(sigma+1e-5) * self.weight
        # return torch.nn.functional.sigmoid(out)   
        return out



class Mlp(nn.Module):
    def __init__(self, in_features, out_features=None, hidden_features=None, act_func=nn.GELU, drop=0., bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features*2
        self.fc1 = nn.Linear(in_features, hidden_features,bias=bias)
        self.act1 = act_func()
        self.fc2 = nn.Linear(hidden_features, out_features,bias=bias)
        self.drop = nn.Dropout(drop)
        self.act2 = act_func()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.fc2(x)
        # x = self.act2(x)
        x = self.drop(x)
        return x


class Conv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1), 
                 dilation=(1,1), groups=1, bias=True, dropout=0, norm=None, act_func=None):
        super().__init__()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.norm = norm
        self.act = act_func() if act_func else None
        if norm:
            self.scale = nn.Parameter(torch.tensor(1.))
            self.shift = nn.Parameter(torch.tensor(0.))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.scale*self.norm(x) + self.shift
        if self.act:
            x = self.act(x)
        return x

        
class WTConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, wt_levels=2 ,bias=True, 
                 dropout=0, norm=None, act_func=None):
        super().__init__()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
        self.conv = WTConv2d(in_channels, out_channels, kernel_size, stride, bias, wt_levels=wt_levels)
        self.norm = norm
        self.act = act_func() if act_func else None
        if norm:
            self.scale = nn.Parameter(torch.tensor(1.))
            self.shift = nn.Parameter(torch.tensor(0.))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.scale*self.norm(x) + self.shift
        if self.act:
            x = self.act(x)
        return x



class DeConv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=4, kernel_size=(3,3), groups=1,
                 bias=True, dropout=0, norm=None, act_func=None):
        super().__init__()

        padding_w = (kernel_size[1] - ratio + 1) // 2
        padding_w = max(0, padding_w)
        output_padding_w = ratio - kernel_size[1] + 2 * padding_w
        
        if not (0 <= output_padding_w < ratio):
            raise ValueError(f"output_padding {output_padding_w} 必须满足 0 <= output_padding < {ratio}")

        self.dropout = nn.Dropout3d(dropout, inplace=False) if dropout > 0 else None
        self.trans_conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(ratio,ratio),
            padding=(padding_w,padding_w),
            output_padding=(output_padding_w,output_padding_w),
            groups=groups,
            bias=bias,
        )

        self.norm = norm if norm else None
        self.act = act_func() if act_func else None
        if norm:
            self.scale = nn.Parameter(torch.tensor(1.))
            self.shift = nn.Parameter(torch.tensor(0.))
    
    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.trans_conv(x)
        if self.norm:
            x = self.scale*self.norm(x) + self.shift
        if self.act:
            x = self.act(x)
        return x



class Swish(nn.Module):
    def __init__(self,beta_init=1.0):
        super().__init__()
        self.beta=nn.Parameter(torch.tensor(beta_init,dtype=torch.float))
    
    def forward(self, x):
        
        return x*torch.sigmoid(self.beta*x)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2, bias=True):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = Conv2dLayer(dim, hidden_features*2, kernel_size=(1,1), stride=(1,1), padding=(0,0), bias=bias)

        self.dwconv = Conv2dLayer(
            hidden_features*2,
            hidden_features*2,
            kernel_size=(3,3),
            stride=(1,1),
            padding=(1,1),
            groups=hidden_features*2,
            bias=bias,
        )

        self.project_out = Conv2dLayer(hidden_features, dim, kernel_size=(1,1), stride=(1,1), padding=(0,0), bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * nn.functional.sigmoid(x2)
        x = self.project_out(x)
        return x

class ConvFFD(nn.Module):
    def __init__(self, dim, bias=True):
        super(ConvFFD,self).__init__()
        self.in_proj = nn.Linear(dim, dim*2, bias=bias)
        self.dw_conv = Conv2dLayer(
            dim*2,
            dim*2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
            groups=dim*2,
        )
        self.out_proj = nn.Linear(dim*2,dim,bias=bias)
        self.act = nn.GELU()

    def forward(self, x):
        b,l,d = x.shape
        h = w = int(math.sqrt(l))
        x = self.in_proj(x)
        x = self.dw_conv(x.view(b,h,w,2*d).permute(0,3,1,2)).flatten(2).transpose(1,2)
        x = self.act(x)
        x = self.out_proj(x)
        return x
        
        
        
class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=2, in_channels=3, embed_dim=256, kernel=6, num_frames=5, target_frames=3,  wt_levels=2, ls_init_value=1, act=nn.GELU,InstanceNorm=True):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        # print(patch_size)
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.target_frames = target_frames

        # self.mlp = Mlp(embed_dim)
        # self.act = act()
        self.gamma = nn.Parameter(ls_init_value * torch.ones(embed_dim)) if ls_init_value is not None else None

        self.conv1 = nn.Sequential(
            WTConvLayer(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel,
                stride=1,
                bias=False,
                wt_levels=wt_levels,
                act_func=nn.GELU,
                # norm=nn.GroupNorm(in_channels, in_channels),
            )
        )
           
        self.conv2 = nn.Sequential(
            Conv2dLayer(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=(3, 3),
                # kernel_size=(5, 5),
                stride=(1, 1),
                padding=(1, 1),
                # padding=(2, 2),
                groups=1,
                bias=False,
                act_func=nn.GELU,
                # norm=nn.GroupNorm(4, embed_dim),
            ),
        )

        self.conv3 = nn.Sequential(
            WTConvLayer(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=kernel,
                stride=1,
                bias=False,
                wt_levels=wt_levels,
                # act_func=nn.GELU,
                norm = nn.InstanceNorm2d(embed_dim) if InstanceNorm else nn.GroupNorm(4, embed_dim)
                # norm = nn.InstanceNorm2d(embed_dim)
            ),
        )

        self.alpha1=nn.Parameter(torch.tensor(1,dtype=torch.float))
        self.beta1=nn.Parameter(torch.tensor(1,dtype=torch.float))
        self.alpha2=nn.Parameter(torch.tensor(1,dtype=torch.float))
        self.beta2=nn.Parameter(torch.tensor(1,dtype=torch.float))

        # self.norm = norm = nn.GroupNorm(embed_dim//4, embed_dim)
        # self.scale = nn.Parameter(torch.tensor(1.))
        # self.shift = nn.Parameter(torch.tensor(0.))
    

    def forward(self, x):
        b, l, d = x.shape
        h = w = int(math.sqrt(l))
        c = self.embed_dim
        x = x.view(b, h, w, d).permute(0,3,1,2)
        res = x[:,-1,:,:]
        
        x = self.alpha1*self.conv1(x) + self.beta1*x
        shortcut = self.conv2(x)
        x = self.alpha2*self.conv3(shortcut) + self.beta2*shortcut
        if self.gamma is not None:
            x = x.mul(self.gamma.view(1, -1, 1, 1))

        x = x.flatten(2).transpose(1, 2)

        return x, res



class SimplePatchEmbed(nn.Module):
# class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=2, in_channels=3, embed_dim=256, kernel=6, num_frames=5, target_frames=3,  wt_levels=2, ls_init_value=1, act=nn.GELU,InstanceNorm=True):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.target_frames = target_frames
        # self.norm = nn.BatchNorm3d(1)

        self.conv = Conv2dLayer(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=(kernel, kernel),
            stride=(1,1),  # 修改步幅
            padding=((kernel - 1) // 2, (kernel - 1) // 2), 
            groups=1,
            bias=False,
            norm = nn.InstanceNorm2d(embed_dim) if InstanceNorm else nn.GroupNorm(4, embed_dim)
        )

    def forward(self, x):
        b, l, d = x.shape
        h = w = int(math.sqrt(l))
        c = self.embed_dim
        x = x.view(b, h, w, d).permute(0,3,1,2)
        res = x[:,-1,:,:]
        
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        return x, res 


class WTLayer(nn.Module):
    def __init__(self, this_dim=128, next_dim=256, kernel=5, bias=True, wt_levels=2, ls_init_value=1, act=nn.GELU, if_res=False, InstanceNorm=True) :
        super().__init__()
        self.next_dim = next_dim
        norm_group = 8 if if_res else 4
        # norm_group = 4
        self.wtconv = WTConvLayer(
            in_channels=this_dim,
            out_channels=this_dim,
            kernel_size=kernel,
            stride=1,
            bias=bias,
            wt_levels=wt_levels,
            norm = nn.InstanceNorm2d(this_dim) if InstanceNorm else nn.GroupNorm(norm_group, this_dim)
            # norm = nn.InstanceNorm2d(this_dim)
            # act_func=nn.GELU,
        )
        
        self.conv = Conv2dLayer(
            in_channels=this_dim,
            out_channels=next_dim,
            kernel_size=3,
            padding=1,
            # kernel_size=5,
            # padding=2,
            stride=1,
            bias=True,
            # norm=nn.GroupNorm(4, next_dim),
            act_func=nn.GELU,
        )

        self.mlp = Mlp(this_dim)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(this_dim)) if ls_init_value is not None else None

        self.alpha=nn.Parameter(torch.tensor(1,dtype=torch.float))
        self.beta=nn.Parameter(torch.tensor(1,dtype=torch.float))
        self.gama1=nn.Parameter(torch.tensor(1,dtype=torch.float))
        self.gama2=nn.Parameter(torch.tensor(1,dtype=torch.float))
        self.gama3=nn.Parameter(torch.tensor(1,dtype=torch.float))
        self.gama4=nn.Parameter(torch.tensor(1,dtype=torch.float))

        # self.norm = norm=nn.GroupNorm(this_dim//4, this_dim)
        # self.scale = nn.Parameter(torch.tensor(1.))
        # self.shift = nn.Parameter(torch.tensor(0.))
    def forward(self, x, residual=None, features=None):
        next_dim = self.next_dim

        if residual is not None:
            x = torch.cat((self.gama1*x, self.gama2*residual), dim=-1)
            if features is not None:
                torch.cat((self.gama3 * features, self.gama4 * features), dim=-1)
        elif features is not None:
                x = x + self.gama3 * features
        
        b, l, d = x.shape
        h = w = int(math.sqrt(l))
        
        x = x.view(b, h, w, d).permute(0, 3, 1, 2)  # (B, D, H, W)

        shortcut=x
        x = self.alpha*self.wtconv(x) + self.beta*shortcut
        x = self.mlp(x.permute(0,2,3,1)).permute(0,3,1,2)
        if self.gamma is not None:
            x = x.mul(self.gamma.view(1, -1, 1, 1))
            
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).view(b, h * w, next_dim)
        
        return x


class LayerToLayer(nn.Module):
# class WTLayer(nn.Module):
    def __init__(self, this_dim=128, next_dim=256, kernel=5, bias=True, wt_levels=2, ls_init_value=1, act=nn.GELU, if_res=False, InstanceNorm=True) :
        super().__init__()
        self.next_dim = next_dim
        norm_group = 8 if if_res else 4
        self.conv = Conv2dLayer(
            in_channels=this_dim,
            out_channels=next_dim,
            kernel_size=(3,3),
            padding=(1,1),
            stride=1,
            bias=bias,
            norm = nn.InstanceNorm2d(next_dim) if InstanceNorm else nn.GroupNorm(norm_group, next_dim),
            act_func=nn.GELU,
        )
        self.gama1=nn.Parameter(torch.tensor(1,dtype=torch.float))
        self.gama2=nn.Parameter(torch.tensor(1,dtype=torch.float))
        self.gama3=nn.Parameter(torch.tensor(1,dtype=torch.float))
        self.gama4=nn.Parameter(torch.tensor(1,dtype=torch.float))
    def forward(self, x, residual=None, features=None):
        next_dim = self.next_dim

        if residual is not None:
            x = torch.cat((self.gama1*x, self.gama2*residual), dim=-1)
            if features is not None:
                torch.cat((self.gama3 * features, self.gama4 * features), dim=-1)
        elif features is not None:
                x = x + self.gama3 * features
        
        b, l, d = x.shape
        h = w = int(math.sqrt(l))
        x = x.view(b, h, w, d).permute(0, 3, 1, 2)  # (B, D, H, W)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).view(b, h * w, next_dim)

        return x

class DownSample(nn.Module):
    def __init__(self, dim=256, kernel=3, ratio=2) :
        super().__init__()
        self.ratio = ratio
        self.dim=dim
        self.max_pool = nn.MaxPool2d(
            kernel_size=ratio,
            stride=ratio,
            padding=0
        )
        
    def forward(self, x):
        ratio = self.ratio
        img = x
        b, l, d = img.shape
        h = w = int(math.sqrt(l))       
        img = img.view(b, h, w, d).permute(0, 3, 1, 2)  # (B, D, H, W)
        img = self.max_pool(img)
        h_new, w_new = h//ratio, w//ratio
        x = img.permute(0, 2, 3, 1).view(b, h_new * w_new, d)
        return x


class UpSample(nn.Module):
    def __init__(self, dim=128, kernel=3, ratio=2, bias=True):
        super().__init__()
        self.ratio = ratio
        self.trans_conv = DeConv2dLayer(
            in_channels=dim,
            out_channels=dim,
            ratio=ratio,
            kernel_size=(kernel, kernel),
            bias = bias,
            act_func=None,
        )

        # self.beta = nn.Parameter(torch.tensor(1.))
        # self.gama = nn.Parameter(torch.tensor(1.))

    def forward(self, x):
        img = x
        b, l, d = img.shape
        h = w = int(math.sqrt(l))  # 原始空间尺寸
        img = img.view(b, h, w, d).permute(0, 3, 1, 2)  # [b, d, h, w]
        
        # img = self.conv(img)
        # img = self.pixel_shuffle(img)

        img = self.trans_conv(img)
        
        new_h, new_w = h * self.ratio, w * self.ratio
        img = img.flatten(2).transpose(1,2)
        
        return img

# This is to extend our grateful acknowledgement to Senior Sihao Zhao.
class IntensityGate(nn.Module):
    def __init__(self, threshold=0.):
        super().__init__()
        self.threshold=nn.Parameter(torch.tensor(threshold,dtype=torch.float))
        self.enhance=nn.Parameter(torch.tensor(1,dtype=torch.float))
        self.act=nn.SiLU()
        # self.act=nn.Sigmoid()
        
    def forward(self, x):
        return self.act(self.enhance*(x - self.threshold))


class Channel_Att_Bridge(nn.Module):
    def __init__(self, c_list=[8, 16, 32, 64, 128, 256], split_att='fc'):
        super().__init__()
        c_list_sum = sum(c_list)
        self.split_att = split_att
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=True)
        self.att1 = nn.Linear(c_list_sum, c_list[0]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[0], 1)
        self.att2 = nn.Linear(c_list_sum, c_list[1]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[1], 1)
        self.att3 = nn.Linear(c_list_sum, c_list[2]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[2], 1)
        self.att4 = nn.Linear(c_list_sum, c_list[3]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[3], 1)
        self.att5 = nn.Linear(c_list_sum, c_list[4]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[4], 1)
        self.att6 = nn.Linear(c_list_sum, c_list[5]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[5], 1)
        self.att7 = nn.Linear(c_list_sum, c_list[6]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[6], 1)

        
        self.sigmoid1 = IntensityGate()
        # self.sigmoid2 = IntensityGate()
        # self.sigmoid3 = IntensityGate()
        # self.sigmoid4 = IntensityGate()
        # self.sigmoid5 = IntensityGate()
        # self.sigmoid6 = IntensityGate()
        # self.sigmoid7 = IntensityGate()

    def forward(self, t):
        for i in range(len(t)):
            b, l, d  = t[i].shape
            h,w = int(math.sqrt(l)), int(math.sqrt(l))
            t[i] = t[i].view(b,h,w,d).permute(0,3,1,2)
  
        att = torch.cat((self.avgpool(t[0]),
                         self.avgpool(t[1]),
                         self.avgpool(t[2]),
                         self.avgpool(t[3]),
                         self.avgpool(t[4]),
                         self.avgpool(t[5]),
                         self.avgpool(t[6])
                         ), dim=1)
        att = self.get_all_att(att.squeeze(-1).transpose(-1, -2))
        if self.split_att != 'fc':
            att = att.transpose(-1, -2)
        att1 = self.sigmoid1(self.att1(att))
        att2 = self.sigmoid1(self.att2(att))
        att3 = self.sigmoid1(self.att3(att))
        att4 = self.sigmoid1(self.att4(att))
        att5 = self.sigmoid1(self.att5(att))
        att6 = self.sigmoid1(self.att6(att))
        att7 = self.sigmoid1(self.att7(att))
        if self.split_att == 'fc':
            att1 = att1.transpose(-1, -2).unsqueeze(-1).expand_as(t[0])
            att2 = att2.transpose(-1, -2).unsqueeze(-1).expand_as(t[1])
            att3 = att3.transpose(-1, -2).unsqueeze(-1).expand_as(t[2])
            att4 = att4.transpose(-1, -2).unsqueeze(-1).expand_as(t[3])
            att5 = att5.transpose(-1, -2).unsqueeze(-1).expand_as(t[4])
            att6 = att6.transpose(-1, -2).unsqueeze(-1).expand_as(t[5])
            att7 = att7.transpose(-1, -2).unsqueeze(-1).expand_as(t[6])
        else:
            att1 = att1.unsqueeze(-1).expand_as(t[0])
            att2 = att2.unsqueeze(-1).expand_as(t[1])
            att3 = att3.unsqueeze(-1).expand_as(t[2])
            att4 = att4.unsqueeze(-1).expand_as(t[3])
            att5 = att5.unsqueeze(-1).expand_as(t[4])
            att6 = att6.unsqueeze(-1).expand_as(t[5])
            att7 = att7.unsqueeze(-1).expand_as(t[6])

        t[0] = t[0].flatten(2).transpose(1,2)
        t[1] = t[1].flatten(2).transpose(1,2)
        t[2] = t[2].flatten(2).transpose(1,2)
        t[3] = t[3].flatten(2).transpose(1,2)
        t[4] = t[4].flatten(2).transpose(1,2)
        t[5] = t[5].flatten(2).transpose(1,2)
        t[6] = t[6].flatten(2).transpose(1,2)
        attn = {}
        attn[0] = att1.flatten(2).transpose(1,2)
        attn[1] = att2.flatten(2).transpose(1,2)
        attn[2] = att3.flatten(2).transpose(1,2)
        attn[3] = att4.flatten(2).transpose(1,2)
        attn[4] = att5.flatten(2).transpose(1,2)
        attn[5] = att6.flatten(2).transpose(1,2)
        attn[6] = att7.flatten(2).transpose(1,2)

        return attn



class EncoderToDecoder(nn.Module):
    def __init__(self, embed_dim=256, InstanceNorm=True):
        super().__init__()
        self.conv13pool= Conv2dLayer(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=(1,3),
            stride=(1,1),
            padding=(0,1),
            bias=True,
            groups=embed_dim//4,
            act_func=nn.GELU,
        )

        self.ffd13=Conv2dLayer(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=embed_dim,
            bias=True,
            # act_func=nn.GELU,
        )
        self.act_func13=IntensityGate()

        self.conv31pool= Conv2dLayer(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=(3,1),
            stride=(1,1),
            padding=(1,0),
            bias=True,
            groups=embed_dim//4,
            act_func=nn.GELU,
        )
        self.ffd31=Conv2dLayer(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=embed_dim,
            bias=True,
            # act_func=nn.GELU,
        )
        self.act_func31=IntensityGate()
     
        self.conv33pool= Conv2dLayer(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=(3,3),
            stride=(1,1),
            padding=(1,1),
            bias=True,
            groups=embed_dim//4,
            act_func=nn.GELU,
        )
        self.ffd33=Conv2dLayer(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=embed_dim,
            bias=True,
            # act_func=nn.GELU,
        )
        self.act_func33=IntensityGate()

        self.max_pool_13 = nn.MaxPool2d(
            kernel_size=(1,3),
            stride=(1,1),
            padding=(0,1)
        )

        self.avg_pool_13 = nn.AvgPool2d(
            kernel_size=(1,3),
            stride=(1,1),
            padding=(0,1)
        )

        
        self.max_pool_31 = nn.MaxPool2d(
            kernel_size=(3,1),
            stride=(1,1),
            padding=(1,0)
        )

        self.avg_pool_31 = nn.AvgPool2d(
            kernel_size=(3,1),
            stride=(1,1),
            padding=(1,0)
        )

        self.max_pool_33 = nn.MaxPool2d(
            kernel_size=(3,3),
            stride=(1,1),
            padding=(1,1)
        )

        self.avg_pool_33 = nn.AvgPool2d(
            kernel_size=(3,3),
            stride=(1,1),
            padding=(1,1)
        )

        self.conv33= Conv2dLayer(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=(3,3),
            stride=(1,1),
            padding=(1,1),
            bias=True,
            groups=embed_dim//4,
            act_func=nn.GELU,
            # norm=BiasFree_LayerNorm(embed_dim),
        )

        self.ffd=FeedForward(dim=embed_dim, bias=True)
        self.act=IntensityGate()
        self.norm=nn.InstanceNorm2d(embed_dim) if InstanceNorm else nn.GroupNorm(4, embed_dim)
        # self.norm = nn.InstanceNorm2d(embed_dim)
        
        self.alpha1=nn.Parameter(torch.tensor(0.33,dtype=torch.float))
        self.alpha2=nn.Parameter(torch.tensor(0.33,dtype=torch.float))
        self.alpha3=nn.Parameter(torch.tensor(0.33,dtype=torch.float))
        self.gama=nn.Parameter(torch.tensor(1,dtype=torch.float))

        self.gamma = nn.Parameter(1 * torch.ones(embed_dim))

        self.mlp = ConvFFD(embed_dim, bias=True)

        self.scale = nn.Parameter(torch.tensor(1.))
        self.shift = nn.Parameter(torch.tensor(0.))

        # self.beta1 = nn.Parameter(torch.tensor(1.))
        # self.beta2 = nn.Parameter(torch.tensor(1.))

    def forward(self, x, res ):
        # print(x.shape)
        b, l, d = x.shape
        h, w = int(math.sqrt(l)), int(math.sqrt(l))

        x = x.view(b,h,w,d).permute(0,3,1,2)
        x = self.scale*self.norm(self.act(x + self.gama * res.view(b,h,w,d).permute(0,3,1,2))) + self.shift 

        x31_pool = self.max_pool_31(x) + self.avg_pool_31(x)
        x13_conv = self.conv13pool(x31_pool)
        x1 = x * x13_conv
        x1=self.ffd13(x1)
        x1=self.act_func13(x1)
        
        x13_pool = self.max_pool_13(x) + self.avg_pool_13(x)
        x31_conv = self.conv31pool(x13_pool)
        x2 = x * x31_conv
        x2=self.ffd13(x2)
        x2=self.act_func13(x2)

        x33_pool = self.max_pool_33(x) + self.avg_pool_33(x)
        x33_conv = self.conv33pool(x33_pool)
        x3 = x * x33_conv
        x3=self.ffd33(x3)
        x3=self.act_func33(x3)
 
        x_pool = self.alpha1*x1 + self.alpha2*x2 + self.alpha3*x3
        if self.gamma is not None:
            x_pool = x_pool.mul(self.gamma.view(1, -1, 1, 1))
        # x_pool = self.conv33(x_pool)

        x_pool=self.ffd(x_pool)
        x_final = self.mlp(x_pool.flatten(2).transpose(1,2))


        return x_final




class OutProj(nn.Module):
    def __init__(self, num_frames=3, embed_dim=256, img_size=[256,256], act_func=Swish,  wt_levels=2, ls_init_value=1,out_expand=2, InstanceNorm=True):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.activation = act_func
        
        self.wtconv = WTConvLayer(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=5,
            stride=1,
            bias=False,
            wt_levels=3,
            act_func=nn.GELU,
            norm=nn.InstanceNorm2d(embed_dim) if InstanceNorm else nn.GroupNorm(4, embed_dim)
            # norm = nn.InstanceNorm2d(embed_dim)
        )

        self.conv = nn.Sequential(
            Conv2dLayer(
                in_channels=embed_dim,
                out_channels=embed_dim*out_expand,
                kernel_size=(3,3),
                # kernel_size=(5,5),
                stride=(1,1),
                padding=(1,1),
                # padding=(2,2),
                bias=False,
                act_func=nn.GELU,
                # norm = nn.GroupNorm(4, embed_dim)
            ),
            Conv2dLayer(
                in_channels=embed_dim*out_expand,
                out_channels=num_frames,
                kernel_size=(1,1),
                stride=(1,1),
                padding=(0,0),
                bias=False,
                act_func=nn.GELU,
            ),
        )

        self.conv2 = Conv2dLayer(
            in_channels=num_frames,
            out_channels=num_frames,
            kernel_size=3,
            stride=1,
            bias=False,
            act_func=self.activation,
        )
        # self.out_wtconv = WTConvLayer(
        #     in_channels=num_frames,
        #     out_channels=num_frames,
        #     kernel_size=3,
        #     stride=1,
        #     bias=False,
        #     wt_levels=3,
        #     # act_func=self.activation,
        # )
        self.alpha1=nn.Parameter(torch.tensor(1.,dtype=torch.float))
        self.alpha2=nn.Parameter(torch.tensor(1.,dtype=torch.float))

        self.gamma = nn.Parameter(ls_init_value * torch.ones(embed_dim)) if ls_init_value is not None else None

        self.alpha=nn.Parameter(torch.tensor(1.,dtype=torch.float))
        self.beta=nn.Parameter(torch.tensor(1.,dtype=torch.float))

        # self.norm = nn.GroupNorm(4, embed_dim)
        # self.scale = nn.Parameter(torch.tensor(1.))
        # self.shift = nn.Parameter(torch.tensor(0.))
    
    def forward(self, x, residual):
        h = self.img_size[0]
        w = self.img_size[1]
        embed_dim = self.embed_dim
        b, l, d = x.shape
            
        x = x.view(b,h,w,d).permute(0,3,1,2)
        # x = self.scale*self.norm(x)+self.shift
        
        shortcut=x
        x = self.alpha*self.wtconv(x) + self.beta*shortcut
        if self.gamma is not None:
            x = x.mul(self.gamma.view(1, -1, 1, 1))
            
        x = self.conv(x)

        if residual is not None:
            residual = residual.unsqueeze(1)
            x = self.alpha1*x+self.alpha2*residual
        x = self.conv2(x)

        return x

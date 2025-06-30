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

from timm.layers import DropPath, to_2tuple
from timm.models import register_model
from timm.models.vision_transformer import _load_weights

import math

from models.WTConv2d import WTConv2d

# from mamba.mamba_ssm.modules.mamba2 import Mamba2
from .ADNssd import Mamba2, StandardAttention
# from .Vssd import Mamba2, StandardAttention
from models.MLA import *

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
    # from mamba.mamba_ssm.ops.triton.layernorm_gated import RMSNorm, layernorm_fn, rmsnorm_fn

except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from .model_untils import *
import gc

def print_memory_usage():
    # 获取当前 GPU 的显存使用情况
    allocated_memory = torch.cuda.memory_allocated() / (1024 * 1024*1024)  # 转换为 MB
    reserved_memory = torch.cuda.memory_reserved() / (1024 * 1024*1024)  # 转换为 MB
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024*1024)  # 转换为 MB
    free_memory = total_memory - reserved_memory

    print(f"Total GPU Memory: {total_memory:.2f} gb")
    print(f"Allocated GPU Memory: {allocated_memory:.2f} gb")
    print(f"Reserved GPU Memory: {reserved_memory:.2f} gb")
    print(f"Free GPU Memory: {free_memory:.2f} gb")
    print()


class Block(nn.Module):
    def __init__(
            self, dim, out_dim, mixer, norm_layer=BiasFree_LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0., drop=0.,  patches_resolution=[64, 64], mlp_ratio=4, num_layers=1, act_layer=nn.SiLU, attn=False
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.dim=dim
        self.out_dim=out_dim
        self.num_layers = num_layers 

        self.alpha1=nn.Parameter(torch.tensor(1,dtype=torch.float))
        self.alpha2=nn.Parameter(torch.tensor(1,dtype=torch.float))
        self.alpha3=nn.Parameter(torch.tensor(1,dtype=torch.float))
        self.alpha4=nn.Parameter(torch.tensor(1,dtype=torch.float))
        self.beta1=nn.Parameter(torch.ones(num_layers))
        self.beta2=nn.Parameter(torch.ones(num_layers))
        self.beta3=nn.Parameter(torch.ones(num_layers))
        self.beta4=nn.Parameter(torch.ones(num_layers))

        # self.res_conv = nn.Conv2d(2*this_dim, this_dim, kernel_size=1)
        # norm_group = 16 if if_res else 8
        
        self.mixer_layers = nn.ModuleList([
            mixer() for _ in range(num_layers)
        ])
        
        self.drop_path_layers = nn.ModuleList([
            DropPath(drop_path) if drop_path > 0. else nn.Identity() 
            for _ in range(num_layers)
        ])

        self.norm1_layers = nn.ModuleList([
            norm_layer(dim) for _ in range(num_layers)
            # nn.BatchNorm2d(dim) for _ in range(num_layers)
        ])
        
        # self.norm1 = norm_layer(dim)
        self.ffns = nn.ModuleList([
            FeedForward(dim=dim, ffn_expansion_factor=2, bias=True) for _ in range(num_layers)
        ])

        self.norm2_layers = nn.ModuleList([
            norm_layer(dim) for _ in range(num_layers)
            # nn.BatchNorm2d(dim) for _ in range(num_layers)
        ])

        self.scale1 = nn.ParameterList([nn.Parameter(torch.tensor(1.)) for _ in range(num_layers)])
        self.shift1 = nn.ParameterList([nn.Parameter(torch.tensor(0.)) for _ in range(num_layers)])
        self.scale2 = nn.ParameterList([nn.Parameter(torch.tensor(1.)) for _ in range(num_layers)])
        self.shift2 = nn.ParameterList([nn.Parameter(torch.tensor(0.)) for _ in range(num_layers)])

        self.act=Swish()

        
        
        if self.dim != self.out_dim:
            # self.out_proj = Mlp(dim, out_dim)
            self.out_proj = nn.Linear(dim, out_dim)
            # self.gamma = nn.Parameter(1 * torch.ones(out_dim))
        # else:
        #     self.gamma = nn.Parameter(1 * torch.ones(dim))
        self.gamma = nn.Parameter(1 * torch.ones(dim))

    def forward(
            self, hidden_states: Tensor, residual=None, features=None, inference_params=None,
            use_checkpoint=False
    ):         
        b, l, d = hidden_states.shape # l = t+t*h*w
        h, w = int(math.sqrt(l)), int(math.sqrt(l))

        x = hidden_states

        if residual is not None:
            x = torch.cat((self.alpha1*x,self.alpha2*residual),dim=-1)
            # x = torch.cat((x,residual),dim=-1)
            d = x.shape[-1]
            if features is not None:
                x = x + torch.cat((self.alpha3 * features, self.alpha4 * features), dim=-1)
        elif features is not None:
            x = x + self.alpha3 * features

        for i in range(self.num_layers):
            layer = self.mixer_layers[i]
            norm1 = self.norm1_layers[i]
            drop_path = self.drop_path_layers[i]
            ffn = self.ffns[i]
            norm2 = self.norm2_layers[i]
            scale1 = self.scale1[i]
            shift1 = self.shift1[i]
            scale2 = self.scale2[i]
            shift2 = self.shift2[i]
            beta1=self.beta1[i]
            beta2=self.beta2[i]
            beta3=self.beta1[i]
            beta4=self.beta2[i]
            
            
            x_norm = scale1*norm1(x) + shift1
            # x_norm = norm1(x)
            
            x = beta1*x + beta2*drop_path(layer(x_norm, h, w))
            # x = x + drop_path(layer(x_norm, h, w))
            
            x_norm = scale2*norm2(x) + shift2
            # x_norm = norm2(x)
            
            x = beta3*x + beta4*ffn(x_norm.view(b,h,w,d).permute(0,3,1,2)).flatten(2).permute(0,2,1)
            # x = x + ffn(x_norm.reshape(b,h,w,d).permute(0,3,1,2)).flatten(2).permute(0,2,1)

        x = x.mul(self.gamma.view(1, 1, -1,))
        if self.dim != self.out_dim:
            x = self.out_proj(x)
        
        return x

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)



class Attention(nn.Module):
    def __init__(self,dim, out_dim=None, headdim=4):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim or dim
            
        self.attn_norm1 = BiasFree_LayerNorm(dim)
        self.attn_norm2 = BiasFree_LayerNorm(dim)
        
        self.attn_layer = StandardAttention(dim, heads=dim//headdim, dim_head=headdim, dropout=0.)
        self.attn_mlp = Mlp(dim)
        self.attn_scale1 = nn.Parameter(torch.tensor(1.))
        self.attn_shift1 = nn.Parameter(torch.tensor(0.))
        self.attn_scale2 = nn.Parameter(torch.tensor(1.))
        self.attn_shift2 = nn.Parameter(torch.tensor(0.))

        if self.dim != self.out_dim:
            # self.out_proj = Mlp(dim, out_dim)
            self.out_proj = nn.Linear(dim, out_dim)
        #     self.gamma = nn.Parameter(1 * torch.ones(out_dim))
        # else:
        #     self.gamma = nn.Parameter(1 * torch.ones(dim))
        self.gamma = nn.Parameter(1 * torch.ones(dim))
        
        self.alpha1=nn.Parameter(torch.tensor(1,dtype=torch.float))
        self.alpha2=nn.Parameter(torch.tensor(1,dtype=torch.float))
        self.alpha3=nn.Parameter(torch.tensor(1,dtype=torch.float))
        self.alpha4=nn.Parameter(torch.tensor(1,dtype=torch.float))
        self.beta1=nn.Parameter(torch.tensor(1,dtype=torch.float))
        self.beta2=nn.Parameter(torch.tensor(1,dtype=torch.float))
        self.beta3=nn.Parameter(torch.tensor(1,dtype=torch.float))
        self.beta4=nn.Parameter(torch.tensor(1,dtype=torch.float))
            
    def forward(
            self, hidden_states: Tensor, residual=None, features=None, inference_params=None,
            use_checkpoint=False
    ):         
        b, l, d = hidden_states.shape # l = t+t*h*w
        h, w = int(math.sqrt(l)), int(math.sqrt(l))

        x = hidden_states

        if residual is not None:
            x = torch.cat((self.alpha1*x,self.alpha2*residual),dim=-1)
            # x = torch.cat((x,residual),dim=-1)
            d = x.shape[-1]
            if features is not None:
                x = x + torch.cat((self.alpha3 * features, self.alpha4 * features), dim=-1)
        elif features is not None:
            x = x + self.alpha3 * features

        x_norm = self.attn_scale1*self.attn_norm1(x)+ self.attn_shift1
        # x_norm = self.attn_norm1(x)
        
        x = self.beta1*x + self.beta2*self.attn_layer(x_norm, h, w)
        # x = x + self.attn_layer(x_norm, h, w)
        
        x_norm = self.attn_scale2*self.attn_norm2(x) + self.attn_shift2
        # x_norm = self.attn_norm2(x)
        
        x = self.beta3*x + self.beta4*self.attn_mlp(x_norm)
        # x = x + self.attn_mlp(x_norm)
        x = x.mul(self.gamma.view(1, 1, -1,))
        if self.dim != self.out_dim:
            x = self.out_proj(x)
        
        return x




def create_block(
        d_model,
        out_dim,
        headdim=None,
        ssm_cfg=None,
        num_layers=1,
        norm_epsilon=1e-5,
        drop_path=0.,
        drop=0.,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        layer_idx=None,
        d_state=16,
        device=None,
        dtype=None,
):
    factory_kwargs = {}
    if ssm_cfg is None:
        ssm_cfg = {}
    if headdim is None:
        if d_model<=32:
            headdim=4
        elif d_model<=256:
            headdim=8
        elif d_model<=512:
            headdim=16
        elif d_model<=768:
            headdim=24
        else:
            headdim=32
    else:
        headdim=headdim
    
    mixer = partial(Mamba2, layer_idx=layer_idx, d_model=d_model, headdim=headdim, linear_attn_duality=True, d_state=d_state, **ssm_cfg, **factory_kwargs)
    norm_layer = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
    
    block = Block(
        dim=d_model,
        out_dim=out_dim,
        mixer=mixer,
        num_layers=num_layers,
        norm_layer=norm_layer,
        drop_path=drop_path,
        drop=drop,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block

def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)



class Encoder(nn.Module):
    def __init__(
        self,
        img_size=256,
        depth=[1,1,1],
        embed_dim=[64,128,128],
        headdim=8,
        in_channels=5,
        kernel=[5,4,3],
        ratio=[2,2,2],
        wt_levels=[4,3,2],
        simple_patch=False,
        norm_epsilon=1e-5,
        ssm_cfg=None,
        InstanceNorm=True,
        # bimamba=True
    ):
        super().__init__()
        self.img_size=img_size
        self.depth=depth
        self.embed_dim=embed_dim
        self.in_channels=in_channels
        self.kernel=kernel
        self.ratio=ratio

        if simple_patch:
            self.encoder1 = SimplePatchEmbed(
                img_size=img_size, patch_size=ratio[0],
                in_channels=channels, embed_dim=embed_dim[0], kernel=kernel[0], 
                wt_levels=wt_levels[0],
                num_frames=num_frames,
                InstanceNorm=InstanceNorm,
            )
        else:
            self.encoder1 = PatchEmbed(
                img_size=img_size, patch_size=ratio[0],
                in_channels=in_channels, embed_dim=embed_dim[0], kernel=kernel[0], wt_levels=wt_levels[0],
                # num_frames=num_frames,
                InstanceNorm=InstanceNorm,
            )

        self.down_sample1 = DownSample(dim=embed_dim[0], ratio=ratio[0])

        self.encoder2 = WTLayer(
            this_dim = embed_dim[0],
            next_dim = embed_dim[1],
            kernel=kernel[1],
            wt_levels = wt_levels[1],
            InstanceNorm=InstanceNorm,
        )
        self.down_sample2 = DownSample(dim=embed_dim[1], ratio=ratio[1])

        self.encoder3 = WTLayer(
            this_dim = embed_dim[1],
            next_dim = embed_dim[2],
            kernel=kernel[2],
            wt_levels = wt_levels[2],
            InstanceNorm=InstanceNorm,
        )
        self.down_sample3 = DownSample(dim=embed_dim[2], ratio=ratio[2])

        self.attn = Attention(dim=embed_dim[2],headdim=headdim,)
        
        self.encoder4 = create_block(
                    d_model=embed_dim[2],
                    out_dim=embed_dim[3],
                    headdim=headdim,
                    num_layers=depth[0],
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    # bimamba=bimamba,
                )
        self.down_sample4 = DownSample(dim=embed_dim[3], ratio=ratio[3])

        self.encoder5 = create_block(
                    d_model=embed_dim[3],
                    out_dim=embed_dim[4],
                    headdim=headdim,
                    num_layers=depth[1],
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    # bimamba=bimamba,
                )
        self.down_sample5 = DownSample(dim=embed_dim[4], ratio=ratio[4])

        self.encoder6 = create_block(
                    d_model=embed_dim[4],
                    out_dim=embed_dim[5],
                    headdim=headdim,
                    num_layers=depth[2],
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    # bimamba=bimamba,
                )

        self.attn2=Attention(embed_dim[5], headdim=headdim,)
        # self.attn3=Attention(embed_dim[5], headdim=4)

        self.encoder_layer_residual = {}

    def forward(self, x):
        b,c,h,w = x.shape
        x = x.flatten(2).transpose(1,2)
        # x = x + self.pos_embed
        
        x, res = self.encoder1(x)
        self.encoder_layer_residual[0]=x
        x = self.down_sample1(x)

        x = self.encoder2(x)
        self.encoder_layer_residual[1]=x
        x = self.down_sample2(x)

        x = self.encoder3(x)
        self.encoder_layer_residual[2]=x
        x = self.down_sample3(x)

        x = self.attn(x)
        self.encoder_layer_residual[3]=x
        
        x = self.encoder4(x)
        self.encoder_layer_residual[4]=x
        x = self.down_sample4(x)

        x = self.encoder5(x)
        self.encoder_layer_residual[5]=x
        x = self.down_sample5(x)

        x = self.encoder6(x)
        self.encoder_layer_residual[6]=x

        
        # x = self.encoder4(x)
        # self.encoder_layer_residual[3]=x
        # x = self.down_sample4(x)

        # x = self.encoder5(x)
        # self.encoder_layer_residual[4]=x
        # x = self.down_sample5(x)

        # x = self.encoder6(x)
        # self.encoder_layer_residual[5]=x

        x = self.attn2(x)
        # x = self.attn3(x)

        return x, self.encoder_layer_residual, res


class Decoder(nn.Module):
    def __init__(
        self,
        img_size=256,
        depth=[1,1,1],
        embed_dim=[64,128,128],
        headdim=8,
        refine_dim=[32,32,32],
        kernel=[5,4,3],
        ratio=[2,2,2],
        wt_levels=[4,3,2],
        simple_patch=False,
        norm_epsilon=1e-5,
        ssm_cfg=None,
        InstanceNorm=True,
        # bimamba=True
    ):
        super().__init__()
        self.img_size=img_size
        self.depth=depth
        self.embed_dim=embed_dim
        self.kernel=kernel
        self.ratio=ratio
        
        self.decoder1= create_block(
                    d_model=embed_dim[5],
                    out_dim=embed_dim[4],
                    headdim=headdim,
                    num_layers=depth[2],
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    # bimamba=bimamba,
                )
        self.up_sample1 = UpSample(dim=embed_dim[4], ratio=ratio[4])

        self.decoder2= create_block(
                    d_model=embed_dim[4]*2,
                    # d_model=embed_dim[4],
                    out_dim=embed_dim[3],
                    headdim=headdim,
                    num_layers=depth[1],
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    # bimamba=bimamba,
                )
        self.up_sample2 = UpSample(dim=embed_dim[3], ratio=ratio[3])

        self.decoder3= create_block(
                    d_model=embed_dim[3]*2,
                    # d_model=embed_dim[3],
                    out_dim=embed_dim[2],
                    headdim=headdim,
                    num_layers=depth[0],
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    # bimamba=bimamba,
                )
        self.attn = Attention(embed_dim[2], embed_dim[2], headdim=headdim,)
        self.up_sample3 = UpSample(dim=embed_dim[2], ratio=ratio[2])

        self.decoder4 = WTLayer(
            this_dim = embed_dim[2]*2,
            # this_dim = embed_dim[2],
            next_dim = embed_dim[1],
            kernel=kernel[2],
            wt_levels = wt_levels[2],
            if_res=True,
            # if_res=False,
            InstanceNorm=InstanceNorm,
        )
        self.up_sample4 = UpSample(dim=embed_dim[1], ratio=ratio[1])

        self.decoder5 = WTLayer(
            this_dim = embed_dim[1]*2,
            # this_dim = embed_dim[1],
            next_dim = embed_dim[0],
            kernel=kernel[1],
            wt_levels = wt_levels[1],
            if_res=True,
            # if_res=False,
            InstanceNorm=InstanceNorm,
        )
        self.up_sample5 = UpSample(dim=embed_dim[0], ratio=ratio[0])

        self.decoder6 =WTLayer(
            this_dim = embed_dim[0]*2,
            # this_dim = embed_dim[0],
            next_dim = embed_dim[0],
            kernel=kernel[0],
            wt_levels = wt_levels[0],
            if_res=True,
            # if_res=False,
            InstanceNorm=InstanceNorm,
        )
        
        self.decoder6_s=Conv2dLayer(
            in_channels=embed_dim[0],
            out_channels=refine_dim[0],
            kernel_size=1,
            stride=1,
            padding=0,
            # groups=refine_dim[0]//4
        )

        embed_dim.insert(2, embed_dim[2])
        self.fusion = Channel_Att_Bridge(c_list=embed_dim)

        self.e2ds = nn.ModuleList([
            EncoderToDecoder(
                embed_dim=embed_dim[len(embed_dim)-1-i],
                InstanceNorm=InstanceNorm,
            )
            for i in range(len(embed_dim))
        ])

        self.features={}
        
    def forward(self, x, encoder_layer_residual):
        res_fusion = self.fusion(encoder_layer_residual)
        for i in range(7):
            self.features[i] = self.e2ds[i](x=encoder_layer_residual[6-i], res=res_fusion[6-i])
            
        x = self.up_sample1(self.decoder1(x, features=self.features[0]))

        x = self.up_sample2(self.decoder2(x, residual=encoder_layer_residual[5], features=self.features[1]))

        x = self.decoder3(x, residual=encoder_layer_residual[4], features=self.features[2])
        x = self.attn(x)
        x = self.up_sample3(x)
        
        x = self.up_sample4(self.decoder4(x, residual=encoder_layer_residual[2], features=self.features[4]))

        x = self.up_sample5(self.decoder5(x, residual=encoder_layer_residual[1], features=self.features[5]))

        # x = self.up_sample2(self.decoder2(x,  features=self.features[1]))

        # x = self.decoder3(x, features=self.features[2])
        # x = self.attn(x)
        # x = self.up_sample3(x)
        
        # x = self.up_sample4(self.decoder4(x, features=self.features[4]))

        # x = self.up_sample5(self.decoder5(x,  features=self.features[5]))
        
        x = self.decoder6(x, residual=encoder_layer_residual[0],features=self.features[6])
        
        
        b,l,d=x.shape
        x = self.decoder6_s(x.view(b,256,256,d).permute(0,3,1,2)).flatten(2).transpose(1,2)
        
        return x
        
        
class Refiner(nn.Module):
    def __init__(
        self,
        img_size=256,
        refine_depth=[1,1,1,1],
        refine_dim=[64,128,128],
        wt_levels=[4,3,2],
        out_channels=3,
        refine_headdim=[4,4,4,4],
        norm_epsilon=1e-5,
        out_expand=2,
        ssm_cfg=None,
        InstanceNorm=True,
        # bimamba=True
    ):
        super().__init__()
        self.img_size=img_size
        self.refine_depth=refine_depth

        self.refiner1 = create_block(
                    d_model=refine_dim[0],
                    out_dim = refine_dim[1],
                    num_layers=refine_depth[0],
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    # bimamba=bimamba,
                    headdim=refine_headdim[0],
                    # d_state=16,
                )

        self.refiner2 = create_block(
                    d_model=refine_dim[1],
                    out_dim = refine_dim[2],
                    num_layers=refine_depth[1],
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    # bimamba=bimamba,
                    headdim=refine_headdim[1],
                    # d_state=32,
                )

        self.refiner3 = create_block(
                    d_model=refine_dim[2],
                    out_dim = refine_dim[3],
                    num_layers=refine_depth[2],
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    # bimamba=bimamba,
                    headdim=refine_headdim[2],
                    # d_state=32,
                )

        self.refiner4 = create_block(
                    d_model=refine_dim[3],
                    out_dim = refine_dim[-1],
                    num_layers=refine_depth[3],
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    # bimamba=bimamba,
                    headdim=refine_headdim[3],
                    # d_state=16,
                )

        self.out_proj=OutProj(
            num_frames=out_channels,
            embed_dim=refine_dim[-1],
            img_size=[img_size,img_size],
            wt_levels = wt_levels[0],
            out_expand=out_expand,
            InstanceNorm=InstanceNorm,
        )

    def forward(self, x, res):
        x = self.refiner1(x)
        x = self.refiner2(x)
        x = self.refiner3(x)
        x = self.refiner4(x)
        x = self.out_proj(x, res)
        return x


class VisionMamba(nn.Module):
    def __init__(
            self,
            img_size=256,
            depth=[1,1,1],
            refine_depth=[1,1],
            refine_dim=[32,32,32],
            refine_headdim=[8,4],
            embed_dim=[64,128,128],
            headdim=8,
            channels=5,
            out_channels=3,
            ssm_cfg=None,
            norm_epsilon=1e-5,
            initializer_cfg=None,
            # bimamba=True,
            kernel=[5,4,3],
            ratio=[2,2,2],
            wt_levels=[4,3,2],
            out_expand=2,
            InstanceNorm=True,
            # target_frames=3,
            simple_patch=False,
            e2d=True,
            **kwargs
    ):
        super().__init__()
        
        self.depth = depth

        self.encoder=Encoder(
            img_size=img_size,
            depth=depth,
            embed_dim=embed_dim,
            headdim=headdim,
            in_channels=channels,
            kernel=kernel,
            ratio=ratio,
            wt_levels=wt_levels,
            simple_patch=simple_patch,
            norm_epsilon=norm_epsilon,
            InstanceNorm=InstanceNorm,
            # bimamba=bimamba
        )
        
        self.decoder=Decoder(
            img_size=img_size,
            depth=depth,
            embed_dim=embed_dim,
            headdim=headdim,
            refine_dim=refine_dim,
            kernel=kernel,
            ratio=ratio,
            wt_levels=wt_levels,
            norm_epsilon=norm_epsilon,
            InstanceNorm=InstanceNorm,
            # bimamba=bimamba
        )

        self.refiner=Refiner(
            img_size=img_size,
            refine_depth=refine_depth,
            refine_dim=refine_dim,
            refine_headdim=refine_headdim,
            out_channels=out_channels,
            wt_levels=wt_levels,
            out_expand=out_expand,
            norm_epsilon=norm_epsilon,
            InstanceNorm=InstanceNorm,
            # bimamba=bimamba
        )
        

        # original init
        self.apply(segm_init_weights)
        # trunc_normal_(self.encoder.pos_embed, std=.001)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=len(depth),
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )


    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.encoder_layers + self.decoder_layers + self.refine_layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "temporal_pos_embedding"}

    def get_num_layers(self):
        return len(self.encoder_layers + self.decoder_layers + self.refine_layers)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward(self, x, mask=None):    
        x = x.squeeze(2)
        x, encoder_layer_residual, res = self.encoder(x)
        x = self.decoder(x,encoder_layer_residual)
        out = self.refiner(x, res)
        return out.unsqueeze(2) # (B, T, C, H, W)

def get_scalar_parameters(model):
    scalar_params = []
    for param in model.parameters():
        if param.requires_grad and param.nelement() == 1:  # 检查是否为标量参数
            scalar_params.append(param)
    return scalar_params

def create_vm(
    img_size=256,
    # patch_size=4,
    depth=[1,1,1],
    refine_depth=[1,1,1,1],
    refine_headdim=[4,4,4,4],
    refine_dim=[32,32,32,32],
    embed_dim=[32,64,128,256,512,1024],
    headdim=4,
    channels=3,
    out_channels=3,
    ssm_cfg=None,
    norm_epsilon=1e-6,
    # bimamba=True,
    # video
    kernel=[5,3,3],
    ratio=[2,2,2,2,2,2],
    wt_levels=[3,1,1],
    out_expand=2,
    InstanceNorm=True,
    initializer_cfg=None):
    
    model = VisionMamba(
        img_size=img_size,
        # patch_size=patch_size,
        depth=depth,
        refine_depth=refine_depth,
        refine_headdim=refine_headdim,
        refine_dim=refine_dim,
        embed_dim=embed_dim,
        headdim=headdim,
        channels=channels,
        out_channels=out_channels,
        ssm_cfg=ssm_cfg,
        norm_epsilon=norm_epsilon,
        initializer_cfg=initializer_cfg,
        # bimamba=bimamba,
        # video
        kernel=kernel,
        ratio=ratio,
        wt_levels=wt_levels,
        out_expand=out_expand,
        InstanceNorm=InstanceNorm,
    )
    return model


def videomamba_middle(pretrained=False, **kwargs):
    model = create_vm(
        img_size=256,
        depth=[1,1,1],
        refine_depth=[1,1,1,1],
        refine_headdim=[4,4,4,4],
        refine_dim=[32,32,32,32],
        embed_dim=[32, 64, 128, 256, 512, 1024],
        channels=5,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        # bimamba=True,
        # video
        kernel=[5,3,3],
        ratio=[2,2,2,2,2,2],
        # target_frames=3,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

def create_ADNMUNet(input_frames,output_frames,frame_interval):
    if output_frames>5:
        refine_dim=[32,32,32,32]
    else:
        refine_dim=[32,32,16,16]
        

    if frame_interval < 120/input_frames:
        InstanceNorm=True
        kernel=[5,5,5]
    else:
        InstanceNorm=False
        kernel=[5,3,3]
        
    model = VisionMamba(
        img_size=256,
        depth=[1,1,1],
        refine_depth=[1,1,1,1],
        refine_headdim=[4,4,4,4],
        refine_dim=refine_dim,
        embed_dim=[32, 64, 128, 256, 512, 1024],
        headdim=4,
        channels=input_frames,
        out_channels=output_frames,
        ssm_cfg=None,
        norm_epsilon=1e-6,
        initializer_cfg=None,
        kernel=kernel,
        ratio=[2,2,2,2,2,2],
        wt_levels=[3,2,1],
        out_expand=2,
        InstanceNorm=InstanceNorm,
    )

    return model

if __name__ == '__main__':
    import numpy as np

    num_frames = 5
    img_size = 256
    model = videomamba_middle().cuda()
    
    x = torch.rand(4, 1, 5, img_size, img_size).cuda()
    
    print(model(x).shape)
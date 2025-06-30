import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from einops import rearrange, repeat
import gc
from .model_untils import *
import math
import copy
from .model_untils import *

class tTensor(torch.Tensor):
    @property
    def shape(self):
        shape = super().shape
        return tuple([int(s) for s in shape])


to_ttensor = lambda *args: tuple([tTensor(x) for x in args]) if len(args) > 1 else tTensor(args[0])

class StandardAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., **kwargs):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.inner_dim = inner_dim


    def forward(self, x, H, W):
        # print('attn')
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Mamba2(nn.Module):
    def __init__(
        self,
        d_model,
        d_conv=3, #default to 3 for 2D
        conv_init=None,
        expand=2,
        headdim=8, #default to 64
        ngroups=2,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        bias=False,
        conv_bias=False,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=False, #default to False, for custom implementation
        layer_idx=None,  # Absorb kwarg for general module
        device=None,
        dtype=None,
        linear_attn_duality=True,
        d_state = 16,
        bimamba=True,
        **kwargs
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.bimamba=bimamba
        self.d_model = d_model
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.headdim = headdim
        self.d_state = d_state
        if ngroups == -1:
            ngroups = self.d_inner // self.headdim #equivalent to multi-head attention
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        #convert chunk_size to triton.language.int32
        self.chunk_size = chunk_size#torch.tensor(chunk_size,dtype=torch.int32)
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        self.ssd_positve_dA = kwargs.get('ssd_positve_dA', True) #default to False, ablation for linear attn duality
        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, int(d_in_proj), bias=bias, **factory_kwargs) #

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state


        self.conv_13_x1= nn.Conv2d(
            in_channels=self.d_inner//4,
            out_channels=self.d_inner//4,
            groups=self.d_inner//4,
            bias=conv_bias,
            kernel_size=(1,3),
            padding=(0,1), # top left bottom right
        )
        self.conv_31_x1= nn.Conv2d(
            in_channels=self.d_inner//4,
            out_channels=self.d_inner//4,
            groups=self.d_inner//4,
            bias=conv_bias,
            kernel_size=(3,1),
            padding=(1,0),
        )
        self.conv_13_x2= nn.Conv2d(
            in_channels=self.d_inner//4,
            out_channels=self.d_inner//4,
            groups=self.d_inner//4,
            bias=conv_bias,
            kernel_size=(1,3),
            padding=(0,1),
        )
        self.conv_31_x2= nn.Conv2d(
            in_channels=self.d_inner//4,
            out_channels=self.d_inner//4,
            groups=self.d_inner//4,
            bias=conv_bias,
            kernel_size=(3,1),
            padding=(1,0),
        )

        self.conv_13_bc1= nn.Conv2d(
            in_channels=2 * self.ngroups * self.d_state//4,
            out_channels=2 * self.ngroups * self.d_state//4,
            groups=2 * self.ngroups * self.d_state//4,
            bias=conv_bias,
            kernel_size=(1,3),
            padding=(0,1),
        )
        self.conv_31_bc1= nn.Conv2d(
            in_channels=2 * self.ngroups * self.d_state//4,
            out_channels=2 * self.ngroups * self.d_state//4,
            groups=2 * self.ngroups * self.d_state//4,
            bias=conv_bias,
            kernel_size=(3,1),
            padding=(1,0),
        )
        self.conv_13_bc2= nn.Conv2d(
            in_channels=2 * self.ngroups * self.d_state//4,
            out_channels=2 * self.ngroups * self.d_state//4,
            groups=2 * self.ngroups * self.d_state//4,
            bias=conv_bias,
            kernel_size=(1,3),
            padding=(0,1),
        )
        self.conv_31_bc2= nn.Conv2d(
            in_channels=2 * self.ngroups * self.d_state//4,
            out_channels=2 * self.ngroups * self.d_state//4,
            groups=2 * self.ngroups * self.d_state//4,
            bias=conv_bias,
            kernel_size=(3,1),
            padding=(1,0),
        )
        # self.is_norm_x1=nn.InstanceNorm2d(self.d_inner//4)
        # self.is_norm_x2=nn.InstanceNorm2d(self.d_inner//4)
        # self.is_norm_bc1=nn.InstanceNorm2d(2 * self.ngroups * self.d_state//4)
        # self.is_norm_bc2=nn.InstanceNorm2d(2 * self.ngroups * self.d_state//4)
        
        
        self.conv2d = nn.Conv2d(
            in_channels=conv_dim//2,
            out_channels=conv_dim//2,
            groups=conv_dim//2,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
        # self.conv1d.weight._no_weight_decay = True

        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True

        # self.act = Swish()
        # self.act2 = Swish()
        self.act = nn.SiLU()
        self.act2 = nn.SiLU()
        
        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True


        self.norm = nn.LayerNorm(self.d_inner)
        self.scale = nn.Parameter(torch.tensor(1.))  # 可学习参数
        self.shift = nn.Parameter(torch.tensor(0.))

        #linear attention duality
        self.linear_attn_duality = linear_attn_duality

        # mixer
        self.act_z = nn.SiLU()
        self.conv2d_z = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        self.alpha1 = nn.Parameter(torch.tensor(1,dtype=torch.float))
        self.alpha2 = nn.Parameter(torch.tensor(1,dtype=torch.float))

        self.out_proj = nn.Linear(self.d_inner*2, self.d_model, bias=bias, **factory_kwargs)
        
        self.kwargs = kwargs

    def non_casual_linear_attn(self, x, dt, A, B, C, D, H=None, W=None):
        '''
        non-casual attention duality of mamba v2
        x: (B, L, H, D), equivalent to V in attention
        dt: (B, L, nheads)
        A: (nheads) or (d_inner, d_state)
        B: (B, L, d_state), equivalent to K in attention
        C: (B, L, d_state), equivalent to Q in attention
        D: (nheads), equivalent to the skip connection
        '''

        batch, seqlen, head, dim = x.shape
        dstate = B.shape[2]
        V = x.permute(0, 2, 1, 3) # (B, H, L, D)
        dt = dt.permute(0, 2, 1) # (B, H, L)
        dA = dt.unsqueeze(-1) * A.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)
        if self.ssd_positve_dA: dA = -dA

        V_scaled = V * dA
        K = B.view(batch, 1, seqlen, dstate)# (B, 1, L, D)
        if getattr(self, "__DEBUG__", False):
            A_mat = dA.cpu().detach().numpy()
            A_mat = A_mat.reshape(batch, -1, H, W)
            setattr(self, "__data__", dict(
                dA=A_mat, H=H, W=W, V=V,))

        if B.shape[-1]//self.d_state == 1:
            ## get kv via transpose K and V
            KV = K.transpose(-2, -1) @ V_scaled # (B, H, dstate, D)
            Q = C.view(batch, 1, seqlen, dstate)#.repeat(1, head, 1, 1)
            x = Q @ KV # (B, H, L, D)
            x = x + V * D.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)
            x = x.permute(0, 2, 1, 3).contiguous()  # (B, L, H, D)
        else:
            assert head % self.ngroups == 0
            dstate = dstate // self.ngroups
            K = K.view(batch, 1, seqlen, self.ngroups, dstate).permute(0, 1, 3, 2, 4) # (B, 1, g, L, dstate)
            V_scaled = V_scaled.view(batch, head//self.ngroups, self.ngroups, seqlen, dim) # (B, H//g, g, L, D)
            Q = C.view(batch, 1, seqlen, self.ngroups, dstate).permute(0, 1, 3, 2, 4) # (B, 1, g, L, dstate)

            KV = K.transpose(-2, -1) @ V_scaled # (B, H//g, g, dstate, D)
            x = Q @ KV # (B, H//g, g, L, D)
            V_skip = (V * D.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)).view(batch, head//self.ngroups, self.ngroups, seqlen, dim) # (B, H//g, g, L, D)
            x = x + V_skip # (B, H//g, g, L, D)
            x = x.permute(0, 3, 1, 2, 4).flatten(2, 3).reshape(batch, seqlen, head, dim) # (B, L, H, D)
            x = x.contiguous()

        return x


    def forward(self, u, H, W, seq_idx=None):
        """
        u: (B,C,H,W)
        Returns: same shape as u
        """
        batch, seqlen, dim = u.shape

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)
        initial_states=repeat(self.init_states, "... -> b ...", b=batch) if self.learnable_init_states else None
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)


        z, xBC, dt = torch.split(
            zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
        )
        dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)


        # 2D Convolution
        # xBC = xBC.view(batch, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # xBC = self.act(self.conv2d(xBC))
        # xBC = xBC.permute(0, 2, 3, 1).view(batch, H*W, -1).contiguous()
        # x, B, C = torch.split(
        #     xBC, [self.d_inner ,self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1
        # )
        
        xBC_even_indices = torch.arange(0, self.d_inner + 2 * self.ngroups * self.d_state, 2).to('cuda')
        xBC_odd_indices = torch.arange(1, self.d_inner + 2 * self.ngroups * self.d_state, 2).to('cuda')
        even_part_xBC = torch.index_select(xBC, dim=-1, index=xBC_even_indices).view(batch, H, W, -1).permute(0, 3, 1, 2).contiguous()
        odd_part_xBC = torch.index_select(xBC, dim=-1, index=xBC_odd_indices).view(batch, H, W, -1).permute(0, 3, 1, 2).contiguous()

        even_part_xBC = self.act(self.conv2d(even_part_xBC))

        xBC_oe_indices = torch.arange(0, self.d_inner//2 + self.ngroups * self.d_state, 2).to('cuda')
        xBC_oo_indices = torch.arange(1, self.d_inner//2 + self.ngroups * self.d_state, 2).to('cuda')
        oe_part_xBC = torch.index_select(odd_part_xBC, dim=1, index=xBC_oe_indices).contiguous()
        oo_part_xBC = torch.index_select(odd_part_xBC, dim=1, index=xBC_oo_indices).contiguous()
        x_oe, bc_oe = torch.split(oe_part_xBC, [self.d_inner//4, self.ngroups * self.d_state//2], dim=1)
        x_oo, bc_oo = torch.split(oo_part_xBC, [self.d_inner//4, self.ngroups * self.d_state//2], dim=1)
        # 非对称版
        x_oe = self.act(self.conv_13_x1(self.conv_31_x1(x_oe)))
        x_oo = self.act(self.conv_31_x2(self.conv_13_x2(x_oo)))
        bc_oe = self.act(self.conv_31_bc1(self.conv_13_bc1(bc_oe)))
        bc_oo = self.act(self.conv_13_bc2(self.conv_31_bc2(bc_oo)))

        # # 对称版
        # x_oe = self.conv_13_x1(self.conv_31_x1(x_oe))
        # x_oo = self.conv_31_x2(self.conv_13_x2(x_oo))
        # bc_oe = self.conv_13_bc1(self.conv_31_bc1(bc_oe))
        # bc_oo = self.conv_31_bc2(self.conv_13_bc2(bc_oo))
        
        xbc_oe = torch.cat((x_oe,bc_oe), dim=1)
        xbc_oo = torch.cat((x_oo,bc_oo), dim=1)
        merged_channels = xbc_oe.size(1) * 2
        odd_part_xBC = torch.empty(
            (xbc_oe.size(0), merged_channels, xbc_oe.size(2), xbc_oe.size(3)),
            dtype=xbc_oe.dtype,
            device=xbc_oe.device
        )
        # 按奇偶索引填充通道维度（dim=1）
        odd_part_xBC[:, ::2, :, :] = xbc_oe  # 偶数通道位置填充 xbc_oe
        odd_part_xBC[:, 1::2, :, :] = xbc_oo  # 奇数通道位置填充 xbc_oo
        # odd_part_xBC=self.act2(odd_part_xBC)
        

        even_part_xBC = even_part_xBC.permute(0, 2, 3, 1).view(batch, H*W, -1).contiguous()
        odd_part_xBC = odd_part_xBC.permute(0, 2, 3, 1).view(batch, H*W, -1).contiguous()

        x_even, B_even, C_even = torch.split(even_part_xBC, [self.d_inner//2, self.ngroups * self.d_state//2, self.ngroups * self.d_state//2], dim=-1)
        x_odd, B_odd, C_odd = torch.split(odd_part_xBC, [self.d_inner//2, self.ngroups * self.d_state//2, self.ngroups * self.d_state//2], dim=-1)

        d = dt.shape[-1]
        dt_even_indices = torch.arange(0, d, 2).to('cuda')
        dt_odd_indices = torch.arange(1, d, 2).to('cuda')
        dt_even = torch.index_select(dt, dim=-1, index=dt_even_indices)
        dt_odd = torch.index_select(dt, dim=-1, index=dt_odd_indices)

        dAD = self.D.shape[-1]
        dAD_even_indices = torch.arange(0, dAD, 2).to('cuda')
        dAD_odd_indices = torch.arange(1, dAD, 2).to('cuda')
        A_even = torch.index_select(A, dim=-1, index=dAD_even_indices)
        A_odd = torch.index_select(A, dim=-1, index=dAD_odd_indices)
        D_even = torch.index_select(self.D, dim=-1, index=dAD_even_indices)
        D_odd = torch.index_select(self.D, dim=-1, index=dAD_odd_indices)
        
        z = z.view(batch, H, W, -1).permute(0, 3, 1, 2).contiguous()
        z = self.act(self.conv2d_z(z))
        z = z.permute(0, 2, 3, 1).view(batch, H*W, -1).contiguous()
        
        x_even, x_odd, dt_even, dt_odd, A_even, A_odd, B_even, C_even,B_odd, C_odd = to_ttensor(x_even, x_odd, dt_even, dt_odd, A_even, A_odd, B_even, C_even,B_odd, C_odd)
        
        if self.linear_attn_duality:
            # dt = dt.chunk(2, dim=-1) # (B, L, nheads) -> (B, L, nheads//2)*2
            # A, D = A.chunk(2, dim=-1), self.D.chunk(2,dim=-1) # (nheads) -> (nheads//2)*2
            y1 = self.non_casual_linear_attn(
                rearrange(x_even, "b l (h p) -> b l h p", p=self.headdim),
                dt_even, A_even, B_even, C_even, D_even, H, W
            )
            y2 = self.non_casual_linear_attn(
                rearrange(x_odd, "b l (h p) -> b l h p", p=self.headdim),
                dt_odd, A_odd, B_odd, C_odd, D_odd, H, W
            )
            
            y1 = rearrange(y1, "b l h p -> b l (h p)")
            y2 = rearrange(y2, "b l h p -> b l (h p)")
            
            y = torch.empty_like(y1.repeat(1,1,2))
            y[..., ::2] = y1
            y[..., 1::2] = y2
        
        else:
            if self.bimamba:
                print("bimamba")
                y_forward = mamba_chunk_scan_combined(
                    rearrange(x_even, "b l (h p) -> b l h p", p=self.headdim),
                    dt_even,
                    A_even,
                    to_ttensor(rearrange(B_even, "b l (g n) -> b l g n", g=self.ngroups)),
                    to_ttensor(rearrange(C_even, "b l (g n) -> b l g n", g=self.ngroups)),
                    chunk_size=self.chunk_size, D=D_even, z=None, seq_idx=seq_idx,
                    initial_states=initial_states, **dt_limit_kwargs
                )
                y_backward = mamba_chunk_scan_combined(
                    rearrange(x_odd, "b l (h p) -> b l h p", p=self.headdim).flip(1),
                    dt_odd.flip(1),
                    A_odd,
                    to_ttensor(rearrange(B_odd, "b l (g n) -> b l g n", g=self.ngroups)).flip(1),
                    to_ttensor(rearrange(C_odd, "b l (g n) -> b l g n", g=self.ngroups)).flip(1),
                    chunk_size=self.chunk_size, D=D_odd, z=None, seq_idx=seq_idx,
                    initial_states=initial_states, **dt_limit_kwargs
                )
                y_forward = rearrange(y_forward, "b l h p -> b l (h p)")
                y_backward = rearrange(y_backward, "b l h p -> b l (h p)").flip(1)
                
                y = torch.empty_like(y_forward.repeat(1,1,2))
                y[..., ::2] = y_forward
                y[..., 1::2] = y_backward
                # y = torch.cat([y_forward, y_backward.flip(1)], dim=-2)
            else:
                y = mamba_chunk_scan_combined(
                    to_ttensor(rearrange(x, "b l (h p) -> b l h p", p=self.headdim)),
                    to_ttensor(dt),
                    to_ttensor(A),
                    to_ttensor(rearrange(B, "b l (g n) -> b l g n", g=self.ngroups)),
                    to_ttensor(rearrange(C, "b l (g n) -> b l g n", g=self.ngroups)),
                    chunk_size=self.chunk_size,
                    D=to_ttensor(self.D),
                    z=None,
                    seq_idx=seq_idx,
                    initial_states=initial_states,
                    **dt_limit_kwargs,
                )

        y = self.norm(y)
        # y = self.scale*y + self.shift
        # y = y*z
        y = torch.cat((self.alpha1*y,self.alpha1*z),dim=-1)
        
        out = self.out_proj(y)
        return out




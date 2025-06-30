import torch
from torch import nn
import torch.nn.functional as F
from mamba_ssm import Mamba



class DMFMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
            d_model=input_dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale1 = nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.skip_scale2 = nn.Parameter(torch.ones(1, dtype=torch.float32))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)

        B, C, W, H = x.shape[:4]

        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]

        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        x1 = x_norm
        x2 = x


        group = 8
        channels_per_group = self.input_dim // group
        x2 = x2.view(B, group, channels_per_group, W, H)
        x2 = torch.transpose(x2, 1, 2).contiguous()
        x2 = x2.view(B, -1, W, H)
        x_flat2 = x2.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm2 = self.norm(x_flat2)

        x2 = x_norm2

        x_mamba1 = self.mamba(x1) + x1 * self.skip_scale1
        x_mamba2 = self.mamba(x2) + x2 * self.skip_scale2

        x_mamba = torch.add(x_mamba1, x_mamba2)
        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out


class Channel_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        c_list_sum = sum(c_list) - c_list[-1]
        self.split_att = split_att
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.att1 = nn.Linear(c_list_sum, c_list[0]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[0], 1)
        self.att2 = nn.Linear(c_list_sum, c_list[1]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[1], 1)
        self.att3 = nn.Linear(c_list_sum, c_list[2]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[2], 1)
        self.att4 = nn.Linear(c_list_sum, c_list[3]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[3], 1)
        self.att5 = nn.Linear(c_list_sum, c_list[4]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[4], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, t1, t2, t3, t4, t5):
        att = torch.cat((self.avgpool(t1),
                         self.avgpool(t2),
                         self.avgpool(t3),
                         self.avgpool(t4),
                         self.avgpool(t5)), dim=1)
        att = self.get_all_att(att.squeeze(-1).transpose(-1, -2))
        if self.split_att != 'fc':
            att = att.transpose(-1, -2)
        att1 = self.sigmoid(self.att1(att))
        att2 = self.sigmoid(self.att2(att))
        att3 = self.sigmoid(self.att3(att))
        att4 = self.sigmoid(self.att4(att))
        att5 = self.sigmoid(self.att5(att))
        if self.split_att == 'fc':
            att1 = att1.transpose(-1, -2).unsqueeze(-1).expand_as(t1)
            att2 = att2.transpose(-1, -2).unsqueeze(-1).expand_as(t2)
            att3 = att3.transpose(-1, -2).unsqueeze(-1).expand_as(t3)
            att4 = att4.transpose(-1, -2).unsqueeze(-1).expand_as(t4)
            att5 = att5.transpose(-1, -2).unsqueeze(-1).expand_as(t5)
        else:
            att1 = att1.unsqueeze(-1).expand_as(t1)
            att2 = att2.unsqueeze(-1).expand_as(t2)
            att3 = att3.unsqueeze(-1).expand_as(t3)
            att4 = att4.unsqueeze(-1).expand_as(t4)
            att5 = att5.unsqueeze(-1).expand_as(t5)

        return att1, att2, att3, att4, att5


class Spatial_Att_Bridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_conv2d = nn.Sequential(nn.Conv2d(2, 1, 7, stride=1, padding=9, dilation=3),
                                           nn.Sigmoid())

    def forward(self, t1, t2, t3, t4, t5):
        t_list = [t1, t2, t3, t4, t5]
        att_list = []
        for t in t_list:
            avg_out = torch.mean(t, dim=1, keepdim=True)
            max_out, _ = torch.max(t, dim=1, keepdim=True)
            att = torch.cat([avg_out, max_out], dim=1)
            att = self.shared_conv2d(att)
            att_list.append(att)
        return att_list[0], att_list[1], att_list[2], att_list[3], att_list[4]


class Multi_scale_STAM_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()

        self.catt = Channel_Att_Bridge(c_list, split_att=split_att)
        self.satt = Spatial_Att_Bridge()
        self.satt2 = Spatial_Att_Bridge()
        self.satt3 = Spatial_Att_Bridge()
        self.aphla1 = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        self.aphla2 = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        self.aphla3 = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        self.conv1_3_1 = nn.Conv2d(c_list[0], c_list[0], (1, 3), padding=(0, 1))
        self.conv3_1_1 = nn.Conv2d(c_list[0], c_list[0], (3, 1), padding=(1, 0))
        self.conv1_1_1 = nn.Conv2d(c_list[0], c_list[0], (3, 3), padding=(1, 1))

        self.conv1_3_2 = nn.Conv2d(c_list[1], c_list[1], (1, 3), padding=(0, 1))
        self.conv3_1_2 = nn.Conv2d(c_list[1], c_list[1], (3, 1), padding=(1, 0))
        self.conv1_1_2 = nn.Conv2d(c_list[1], c_list[1], (3, 3), padding=(1, 1))

        self.conv1_3_3 = nn.Conv2d(c_list[2], c_list[2], (1, 3), padding=(0, 1))
        self.conv3_1_3 = nn.Conv2d(c_list[2], c_list[2], (3, 1), padding=(1, 0))
        self.conv1_1_3 = nn.Conv2d(c_list[2], c_list[2], (3, 3), padding=(1, 1))

        self.conv1_3_4 = nn.Conv2d(c_list[3], c_list[3], (1, 3), padding=(0, 1))
        self.conv3_1_4 = nn.Conv2d(c_list[3], c_list[3], (3, 1), padding=(1, 0))
        self.conv1_1_4 = nn.Conv2d(c_list[3], c_list[3], (3, 3), padding=(1, 1))

        self.conv1_3_5 = nn.Conv2d(c_list[4], c_list[4], (1, 3), padding=(0, 1))
        self.conv3_1_5 = nn.Conv2d(c_list[4], c_list[4], (3, 1), padding=(1, 0))
        self.conv1_1_5 = nn.Conv2d(c_list[4], c_list[4], (3, 3), padding=(1, 1))

    def forward(self, t1, t2, t3, t4, t5):
        r1, r2, r3, r4, r5 = t1, t2, t3, t4, t5
        t1_1 = self.conv1_1_1(t1)
        t1_2 = self.conv1_3_1(t1)
        t1_3 = self.conv3_1_1(t1)

        t2_1 = self.conv1_1_2(t2)
        t2_2 = self.conv1_3_2(t2)
        t2_3 = self.conv3_1_2(t2)

        t3_1 = self.conv1_1_3(t3)
        t3_2 = self.conv1_3_3(t3)
        t3_3 = self.conv3_1_3(t3)

        t4_1 = self.conv1_1_4(t4)
        t4_2 = self.conv1_3_4(t4)
        t4_3 = self.conv3_1_4(t4)

        t5_1 = self.conv1_1_5(t5)
        t5_2 = self.conv1_3_5(t5)
        t5_3 = self.conv3_1_5(t5)

        satt1_1, satt2_1, satt3_1, satt4_1, satt5_1 = self.satt(t1_1, t2_1, t3_1, t4_1, t5_1)
        t1_1_1, t2_1_1, t3_1_1, t4_1_1, t5_1_1 = satt1_1 * t1_1, satt2_1 * t2_1, satt3_1 * t3_1, satt4_1 * t4_1, satt5_1 * t5_1

        satt1_2, satt2_2, satt3_2, satt4_2, satt5_2 = self.satt2(t1_2, t2_2, t3_2, t4_2, t5_2)
        t1_1_2, t2_1_2, t3_1_2, t4_1_2, t5_1_2 = satt1_2 * t1_2, satt2_2 * t2_2, satt3_2 * t3_2, satt4_2 * t4_2, satt5_2 * t5_2

        satt1_3, satt2_3, satt3_3, satt4_3, satt5_3 = self.satt3(t1_3, t2_3, t3_3, t4_3, t5_3)
        t1_1_3, t2_1_3, t3_1_3, t4_1_3, t5_1_3 = satt1_3 * t1_3, satt2_3 * t2_3, satt3_3 * t3_3, satt4_3 * t4_3, satt5_3 * t5_3

        r1_, r2_, r3_, r4_, r5_ = self.aphla1 * t1_1_1 + self.aphla2 * t1_1_2 + self.aphla3 * t1_1_3, self.aphla1 * t2_1_1 + self.aphla2 * t2_1_2 + self.aphla3 * t2_1_3, self.aphla1 * t3_1_1 + self.aphla2 * t3_1_2 + self.aphla3 * t3_1_3, self.aphla1 * t4_1_1 + self.aphla2 * t4_1_2 + self.aphla3 * t4_1_3, self.aphla1 * t5_1_1 + self.aphla2 * t5_1_2 + self.aphla3 * t5_1_3
        t1, t2, t3, t4, t5 = r1_ + r1, r2_ + r2, r3_ + r3, r4_ + r4, r5_ + r5


        catt1, catt2, catt3, catt4, catt5 = self.catt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = catt1 * t1, catt2 * t2, catt3 * t3, catt4 * t4, catt5 * t5

        return t1 + r1_, t2 + r2_, t3 + r3_, t4 + r4_, t5 + r5_


class Mamba_UNet(nn.Module):

    def __init__(self, predicted_frames=3, input_frames=5, c_list=[32, 64, 128, 256, 512, 1024],
                 split_att='fc', bridge=True):
        super().__init__()

        self.bridge = bridge

        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_frames, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
        )
        self.encoder4 = nn.Sequential(
            DMFMLayer(input_dim=c_list[2], output_dim=c_list[3])
        )
        self.encoder5 = nn.Sequential(
            DMFMLayer(input_dim=c_list[3], output_dim=c_list[4])
        )
        self.encoder6 = nn.Sequential(
            DMFMLayer(input_dim=c_list[4], output_dim=c_list[5])
        )

        if bridge:
            self.STAM = Multi_scale_STAM_Bridge(c_list, split_att)
            print('Multi_scale_STAM_Bridge was used')

        self.decoder1 = nn.Sequential(
            DMFMLayer(input_dim=c_list[5], output_dim=c_list[4])
        )
        self.decoder2 = nn.Sequential(
            DMFMLayer(input_dim=c_list[4], output_dim=c_list[3])
        )
        self.decoder3 = nn.Sequential(
            DMFMLayer(input_dim=c_list[3], output_dim=c_list[2])
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
        )
        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )

        self.contr1 = nn.Sequential(
            nn.ConvTranspose2d(c_list[3], c_list[3], kernel_size=(2, 2), stride=(2, 2))
        )
        self.contr2 = nn.Sequential(
            nn.ConvTranspose2d(c_list[2], c_list[2], kernel_size=(2, 2), stride=(2, 2))
        )
        self.contr3 = nn.Sequential(
            nn.ConvTranspose2d(c_list[1], c_list[1], kernel_size=(2, 2), stride=(2, 2))
        )
        self.contr4 = nn.Sequential(
            nn.ConvTranspose2d(c_list[0], c_list[0], kernel_size=(2, 2), stride=(2, 2))
        )
        self.contr5 = nn.Sequential(
            nn.ConvTranspose2d(c_list[0], c_list[0], kernel_size=(2, 2), stride=(2, 2))
        )

        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.ebn6 = nn.GroupNorm(4, c_list[5])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])
        self.dbn6 = nn.GroupNorm(4, c_list[0])
        self.dbn7 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], c_list[0], kernel_size=1)

        self.refinement = nn.Sequential(
            DMFMLayer(input_dim=c_list[0], output_dim=c_list[1]),
            DMFMLayer(input_dim=c_list[1], output_dim=c_list[2]),
            DMFMLayer(input_dim=c_list[2], output_dim=c_list[1]),
            DMFMLayer(input_dim=c_list[1], output_dim=c_list[0]),
        )

        self.S1 = nn.Conv2d(c_list[0], predicted_frames, 3, 1, 1)

        self.S = nn.Conv2d(predicted_frames, predicted_frames, 3, 1, 1)
        self.beta = nn.Parameter(torch.tensor(1.0, dtype=torch.float))


    def forward(self, x):
        x = x.squeeze(2)
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out
        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out
        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out
        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out
        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out
        if self.bridge: t1, t2, t3, t4, t5 = self.STAM(t1, t2, t3, t4, t5)
        out = F.gelu(self.ebn6(self.encoder6(out)))
        out5 = F.gelu((self.dbn1(self.decoder1(out))))
        out5 = torch.add(out5, t5)
        out4 = F.gelu(self.contr1(self.dbn2(self.decoder2(out5))))
        out4 = torch.add(out4, t4)
        out3 = F.gelu(self.contr2(self.dbn3(self.decoder3(out4))))
        out3 = torch.add(out3, t3)
        out2 = F.gelu(self.contr3(self.dbn4(self.decoder4(out3))))
        out2 = torch.add(out2, t2)
        out1 = F.gelu(self.contr4(self.dbn5(self.decoder5(out2))))
        out1 = torch.add(out1, t1)
        out0 = F.gelu(self.contr5(self.dbn6(self.final(out1))))
        out0 = F.gelu(self.dbn7(self.refinement(out0)))
        out0 = self.S1(out0)
        out0 = out0 + x[:, -1, ...].unsqueeze(1)
        out0 = self.S(out0)
        out00 = out0 * torch.sigmoid(self.beta * out0)

        out00 = out00.unsqueeze(2)
        return out00

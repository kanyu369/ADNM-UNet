import os
from config import config_root 

from models.ADNMUNet import create_ADNMUNet
from models.ConvLSTM import create_ConvLSTM
from models.TrajGRU import create_TrajGRU
from models.LPTQPN import LPTQPN
from models.TransUnet import create_TransUnet
from models.SmaAt_UNet import SmaAt_UNet
from models.SwinUnet import config, SwinUnet
# from models.MambaUnet import Mamba_UNet

from models.loss import *

import torch.optim.optimizer
import torchvision
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_models(model_name,input_frames,output_frames,frame_interval, dataset):
    if dataset == 'LAPS':
        thresholds = [0.1,0.3,0.5,0.7,0.8]
    elif dataset == 'Shanghai':
        thresholds = [20, 30, 35, 40]
    if model_name == 'ADNMUnet':
        model = create_ADNMUNet(input_frames,output_frames,frame_interval)
        base_LR = 1e-3
        eta_min=5e-7
    
        weight_decay=1e-2
        amsgrad=False
        warmup_epoch=3
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=base_LR,
            betas = (0.9, 0.999),
            eps=1e-9,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        criterion=enRainfallLoss(omega_t=0.57, alpha=0.25, gamma=0.).to(device)
        scheduler_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epoch)
        scheduler_cos = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=eta_min)
        lr_scheduler = optim.lr_scheduler.SequentialLR(optimizer, [scheduler_warmup, scheduler_cos], [warmup_epoch])
        if frame_interval < 120/input_frames: # 根据InstanceNorm决定梯度裁剪强度
            early_stop=3
        else:
            early_stop=5
        
        if_early_stop=True
        ex_lr_scheduler=True
        norm_clip = True
        save_epoch = True
    
    elif model_name == "ConvLSTM":
        model = create_ConvLSTM(output_frames)
        base_LR = 1e-4
        optimizer = torch.optim.Adam(model.parameters(), lr=base_LR)
        criterion = Weighted_mse_mae(thresholds=thresholds).to(device)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15000, 30000], gamma=0.1)
        early_stop=None
        if_early_stop=False
        ex_lr_scheduler=True
        norm_clip = False
        save_epoch = False

    elif model_name == "TrajGRU":
        base_LR = 1e-4
        model = create_TrajGRU(output_frames)
        optimizer = torch.optim.Adam(model.parameters(), lr=base_LR)
        criterion = Weighted_mse_mae(thresholds=thresholds).to(device)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15000, 30000], gamma=0.1)
        early_stop=None
        if_early_stop=False
        ex_lr_scheduler=True
        norm_clip = False
        save_epoch = False

    elif model_name == "LPTQPN":
        model=LPTQPN(inp_channels=input_frames, out_channels=output_frames)
        base_LR=1e-3
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_LR)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30,eta_min=1e-9)
        criterion=RAINlOSS().to(device)
        early_stop=10
        if_early_stop=True
        ex_lr_scheduler=True
        norm_clip = False
        save_epoch = False

    elif model_name == "TransUnet":
        model=create_TransUnet(output_frames)
        base_LR=1e-2
        optimizer = optim.SGD(model.parameters(), lr=base_LR, momentum=0.9, weight_decay=0.0001)
        # criterion=DiceLoss().to(device)
        criterion=RAINlOSS().to(device)
        early_stop=None
        if_early_stop=False
        ex_lr_scheduler=False    
        lr_scheduler=None
        norm_clip = False
        save_epoch = False

    elif model_name=='SmaATUnet':
        model = SmaAt_UNet(n_channels=input_frames, n_classes=output_frames)
        base_LR=1e-2
        optimizer = optim.SGD(model.parameters(), lr=base_LR, momentum=0.9, weight_decay=0.0001)
        criterion=RAINlOSS().to(device)
        early_stop = 30
        if_early_stop=True
        ex_lr_scheduler=False
        lr_scheduler=None
        norm_clip = False
        save_epoch = False

    elif model_name=='SwinUnet':
        model = SwinUnet(config, img_size=config.DATA.IMG_SIZE, num_classes=output_frames)
        base_LR=1e-2
        optimizer = optim.SGD(model.parameters(), lr=base_LR, momentum=0.9, weight_decay=0.0001)
        criterion=RAINlOSS().to(device)
        early_stop=None
        if_early_stop=False
        ex_lr_scheduler=False   
        lr_scheduler=None
        norm_clip = False
        save_epoch = False

    return model, optimizer, criterion, lr_scheduler, if_early_stop, ex_lr_scheduler, norm_clip, save_epoch, base_LR, early_stop
    
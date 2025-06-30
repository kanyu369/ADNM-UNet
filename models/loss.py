import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
import numpy as np
import os
import math
import random

# This is to extend our grateful acknowledgement to Senior Sihao Zhao.
class RainfallLoss(nn.Module):
    def __init__(self, omega_t=0.57, alpha=0.25):
        super(RainfallLoss, self).__init__()
        self.omega_t = omega_t
        self.alpha = alpha
    def forward(self, pred, target):
        #######################-----Dynamic quantile weighted loss(DQWL)----##########################
        w0 = 0.57
        loss_overal1 = torch.sum(((pred >= target)).float() * (1 - w0) * abs(pred - target))
        loss_overal2 = torch.sum(((pred < target)).float() * w0 * abs(pred - target))
        wi = self.alpha * torch.exp(target)
        loss_greater = torch.sum(
            ((pred >= target) & (target >= 0.7)).float() * (1 - self.omega_t) * wi * abs(pred - target))
        loss_less = torch.sum(((pred < target) & (target >= 0.7)).float() * self.omega_t * wi * abs(pred - target))
        num_sample = target.numel()
        total_loss = (loss_overal1 + loss_overal2) / num_sample + (loss_greater + loss_less) / num_sample
        return total_loss


class enRainfallLoss(nn.Module):
    def __init__(self, omega_t=0.57, alpha=0.25, gamma=0.1):
        super(enRainfallLoss, self).__init__()
        self.omega_t = omega_t
        self.alpha = alpha
        self.gamma = gamma  # 新增FN惩罚系数

    def forward(self, pred, target):
        #######################-----Dynamic quantile weighted loss(DQWL)----##########################
        w0 = self.omega_t
        loss_overal1 = torch.sum(((pred >= target)).float() * (1 - w0) * abs(pred - target))
        loss_overal2 = torch.sum(((pred < target)).float() * w0 * abs(pred - target))
        wi = self.alpha * torch.exp(target)
        loss_greater = torch.sum(
            ((pred >= target) & (target >= 0.7)).float() * (1 - self.omega_t) * wi * abs(pred - target))
        loss_less = torch.sum(((pred < target) & (target >= 0.7)).float() * self.omega_t * wi * abs(pred - target))
        
        #######################-----新增FN正则项-----##########################
        # 改进后的高降雨区域预测不足惩罚项（指数增强版）
        fn_penalty = torch.sum(
            ((target >= 0.7) & (pred < target)).float() *      # 高降雨区域的预测不足掩码
            self.gamma *                                        # 基础惩罚系数
            (torch.exp(self.alpha * (target - pred)) - 1.0)    # 指数型误差敏感惩罚
        )

        num_sample = target.numel()
        total_loss = (loss_overal1 + loss_overal2 + loss_greater + loss_less + fn_penalty) / num_sample
        return total_loss



class RAINlOSS(nn.Module):
    def __init__(self):
        super(RAINlOSS, self).__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, x, y):
        loss = self.mse(x, y) + self.mae(x, y)
        return loss



class Weighted_mse_mae(nn.Module):
    def __init__(self, mse_weight=1.0, mae_weight=1.0, NORMAL_LOSS_GLOBAL_SCALE=0.00005, LAMBDA=None, thresholds=[]):
        super().__init__()
        self.NORMAL_LOSS_GLOBAL_SCALE = NORMAL_LOSS_GLOBAL_SCALE
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self._lambda = LAMBDA
        self.thresholds=thresholds

    def forward(self, input, target):
        input = input.transpose(0,1)
        target = target.transpose(0,1)
        balancing_weights = (1, 1, 2, 5, 10, 30)
        weights = torch.ones_like(input) * balancing_weights[0]
        for i, threshold in enumerate(self.thresholds):
            weights = weights + (balancing_weights[i + 1] - balancing_weights[i]) * (target >= threshold).float()
        # weights = weights * mask.float()

        mse = torch.sum(weights * ((input-target)**2), (2, 3, 4))
        mae = torch.sum(weights * (torch.abs((input-target))), (2, 3, 4))
        if self._lambda is not None:
            S, B = mse.size()
            w = torch.arange(1.0, 1.0 + S * self._lambda, self._lambda)
            if torch.cuda.is_available():
                w = w.to(mse.get_device())
            mse = (w * mse.permute(1, 0)).permute(1, 0)
            mae = (w * mae.permute(1, 0)).permute(1, 0)
        return self.NORMAL_LOSS_GLOBAL_SCALE * (self.mse_weight*torch.mean(mse) + self.mae_weight*torch.mean(mae))
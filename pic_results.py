# python -m pic_results
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
from functools import partial
from torch.utils.data import DataLoader

import gc
gc.collect()
torch.cuda.empty_cache()

import os
from config import config_root
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)  # 确保优先从项目根目录查找模块

from train_untils import create_models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_frames = 5
target_frames = 3
batch_size = 4
epochs = 60
best = -10000

dataset = 'LAPS'
dataset = 'Shanghai'

model_name='ADNMUnet'
# model_name='MambaUNet'
# model_name='TrajGRU'
# model_name='ConvLSTM'
# model_name='TransUnet'
# model_name='LPTQPN'
# model_name='SmaATUnet'
# model_name='SwinUnet'

print("-----准备数据集-----")

if dataset == 'LAPS':
    from datasets.LAPS import train_dataset,val_dataset,test_dataset
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,  drop_last=True)
    thresholds = [0.1,0.3,0.5,0.7,0.8]
    input_frames=5
    output_frames=3
    if_cmap=True
    cmap = colors.ListedColormap([ 'white','Lime', 'limegreen', 'green',
                               'darkgreen', 'yellow','gold', 'orange', 'tomato',
                               'red', 'firebrick', 'darkred'])
    BOUNDS = [0, 0.05, 0.15, 0.3, 0.35,0.5,0.65, 0.7, 0.75,0.8,0.85,0.9,1.0]
    PIXEL_SCALE = None
    even_index_only=False
    frame_interval=60

elif dataset == 'Shanghai':
    from datasets.Shanghai import train_dataset,val_dataset,test_dataset
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,  drop_last=True)
    thresholds = [20, 30, 35, 40]
    input_frames=5
    output_frames=20
    PIXEL_SCALE = 90.0
    COLOR_MAP = np.array([
    [0, 0, 0,0],
    [0, 236, 236, 255],
    [1, 160, 246, 255],
    [1, 0, 246, 255],
    [0, 239, 0, 255],
    [0, 200, 0, 255],
    [0, 144, 0, 255],
    [255, 255, 0, 255],
    [231, 192, 0, 255],
    [255, 144, 2, 255],
    [255, 0, 0, 255],
    [166, 0, 0, 255],
    [101, 0, 0, 255],
    [255, 0, 255, 255],
    [153, 85, 201, 255],
    [255, 255, 255, 255]
    ]) / 255
    cmap=None
    even_index_only=True
    
    BOUNDS = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]

    frame_interval=6



def gray2color(image,cmap=None, **kwargs):
    if cmap is None:
        cmap = colors.ListedColormap(COLOR_MAP )
    bounds = BOUNDS
    norm = colors.BoundaryNorm(bounds, cmap.N)
    colored_image = cmap(norm(image))

    return colored_image



def vis_res(pred_seq, gt_seq=None, save_path=None, pic_name=None,
            pixel_scale=None, gray2color=None, cmap=None, gap=10,
            input_seq=None, even_index_only=False):
    """
    添加间隙的气象预测结果可视化函数
    参数说明：
    gap: 图片之间的间隙宽度（像素）
    input_seq: 输入序列 [T, H, W] 或 [T, C, H, W]
    even_index_only: 是否只保存偶数索引的图片（索引0,2,4...）
    """
    # 1. 数据预处理
    def process_seq(seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.detach().cpu().numpy()
        seq = seq.squeeze()
        if pixel_scale is not None:
            seq = (seq * pixel_scale).astype(np.uint8)
        return seq
    
    pred_seq = process_seq(pred_seq)
    if gt_seq is not None:
        gt_seq = process_seq(gt_seq)
    if input_seq is not None:
        input_seq = process_seq(input_seq)
    
    os.makedirs(save_path, exist_ok=True)

    # 2. 选择索引（如果需要只保存偶数索引）
    def select_indices(seq):
        if even_index_only:
            # 选择偶数索引：0,2,4...
            return seq[1::2]
        return seq
    
    pred_seq = select_indices(pred_seq)
    if gt_seq is not None:
        gt_seq = select_indices(gt_seq)


    # 3. 灰度转彩色
    def apply_color(seq):
        if gray2color is not None:
            return np.array([gray2color(seq[i], cmap=cmap) 
                             for i in range(len(seq))])
        return seq
    
    colored_pred = apply_color(pred_seq)
    if gt_seq is not None:
        colored_gt = apply_color(gt_seq)
    if input_seq is not None:
        colored_input = apply_color(input_seq)
    
    # 4. 创建拼接图像（带间隙）
    def create_grid_with_gap(seq, gap_width=gap):
        if len(seq) == 0:
            return None
            
        h, w, c = seq[0].shape
        gap_image = np.ones((h, gap_width, c), dtype=seq[0].dtype)
        
        with_gaps = []
        for i, img in enumerate(seq):
            with_gaps.append(img)
            if i < len(seq) - 1:
                with_gaps.append(gap_image)
        
        return np.concatenate(with_gaps, axis=1)
    
    # 5. 创建并保存所有序列
    grid_pred = create_grid_with_gap(colored_pred)
    if gt_seq is not None:
        grid_gt = create_grid_with_gap(colored_gt)
    
    plt.imsave(os.path.join(save_path, f"{pic_name}.png"), grid_pred)
    if gt_seq is not None:
        plt.imsave(os.path.join(save_path, "gt.png"), grid_gt)
    
    # 6. 如果提供了输入序列，也保存它
    if input_seq is not None:
        grid_input = create_grid_with_gap(colored_input)
        plt.imsave(os.path.join(save_path, "input.png"), grid_input)

color_fn = partial(vis_res, 
                    pixel_scale = PIXEL_SCALE, 
                    # thresholds = thresholds, 
                    gray2color = gray2color)


def vis_res_1b1(pred_seq, save_path=None,
            pixel_scale=None, gray2color=None, cmap=None):

    # 1. 数据预处理
    def process_seq(seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.detach().cpu().numpy()
        seq = seq.squeeze()
        if pixel_scale is not None:
            seq = (seq * pixel_scale).astype(np.uint8)
        return seq
    
    pred_seq = process_seq(pred_seq)
    
    os.makedirs(save_path, exist_ok=True)

    # 3. 灰度转彩色
    def apply_color(seq):
        if gray2color is not None:
            return np.array([gray2color(seq[i], cmap=cmap) 
                             for i in range(len(seq))])
        return seq
    
    colored_pred = apply_color(pred_seq)

    # print(colored_pred.shape)
    for i in range(colored_pred.shape[0]):
        plt.imsave(os.path.join(save_path, f"gt{i}.png"), colored_pred[i])


color_fn = partial(vis_res, 
                    pixel_scale = PIXEL_SCALE, 
                    # thresholds = thresholds, 
                    gray2color = gray2color)


save_path = os.path.join(config_root, 'result_pics2')
save_path = os.path.join(save_path, dataset)
os.makedirs(save_path, exist_ok=True)
params_path = os.path.join(config_root, 'model_params')
params_path = os.path.join(params_path, dataset)
os.makedirs(params_path, exist_ok=True)

print("-----准备模型-----")
model, optimizer, criterion, lr_scheduler, early_stop, ex_lr_scheduler, norm_clip, save_epoch, base_LR, early_stop = create_models(model_name,input_frames,output_frames,frame_interval,dataset)
model.to(device)

model_folder = os.path.join(params_path, model_name)
save_path = os.path.join(save_path, model_name)
os.makedirs(save_path, exist_ok=True)

file_name = f"{model_name}_best.pth"

save_best_path = os.path.join(model_folder, file_name)
if os.path.exists(save_best_path):
    model.load_state_dict(torch.load(save_best_path))
    print(f"成功加载模型参数：{save_best_path}")
else:
    print(f"未找到模型参数文件：{save_best_path}")

print('-----------------create prediction pictures-----------------')
model.eval()
with torch.no_grad():
    cnt=1
    for data in test_dataloader:
        imgs, targets = data[:,:input_frames,:,:,:],data[:,input_frames:,:,:,:]
        imgs_cuda, targets_cuda = imgs.to(torch.float32).to(device), targets.to(torch.float32).to(device)
        outputs = model(imgs_cuda)
        outputs=outputs.cpu().numpy()
        targets_cuda=targets_cuda.cpu().numpy()
        imgs_cuda=imgs_cuda.cpu().numpy()
        for i in range(targets_cuda.shape[0]):
            save_path_cnt_i = os.path.join(save_path, f"{cnt}-{i+1}")
            color_fn(outputs[i],
                     targets_cuda[i],
                     save_path=save_path_cnt_i,
                     pic_name=(model_name),
                     cmap=cmap,
                     even_index_only=even_index_only, 
                    )
        # for i in range (imgs_cuda.shape[0]):
        #     save_path_cnt_i = os.path.join(save_path, f"{cnt}-{i+1}")
        #     # vis_res_1b1(imgs_cuda[i],save_path=save_path_cnt_i,pixel_scale=PIXEL_SCALE,cmap=cmap,gray2color=gray2color)
        # cnt+=1
        


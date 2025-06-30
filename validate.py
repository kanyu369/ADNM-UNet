# python -m validate
import torchvision
from train_untils import create_models
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.nn import SyncBatchNorm

import sys
import gc
gc.collect()
torch.cuda.empty_cache()

import os
from config import config_root

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 4

best = 10000

dataset = 'LAPS'
dataset = 'Shanghai'

model_name='ADNMUnet'
# model_name='TrajGRU'
# model_name='ConvLSTM'
# model_name='TransUnet'
# model_name='LPTQPN'
# model_name='SmaATUnet'
# model_name='SwinUnet'

print("-----准备数据集-----")

if dataset == 'LAPS':
    epochs = 60
    from datasets.LAPS import train_dataset,val_dataset,test_dataset
    from datasets.LAPS_metrics import SimplifiedEvaluator
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,  drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,  drop_last=True)
    thresholds = [0.1,0.3,0.5,0.7,0.8]
    input_frames=5
    output_frames=3
    frame_interval=60
elif dataset == 'Shanghai':
    epochs = 60
    from datasets.Shanghai import train_dataset,val_dataset,test_dataset
    from datasets.Shanghai_metrics import SimplifiedEvaluator
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,  drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,  drop_last=True)
    thresholds = [20, 30, 35, 40]
    input_frames=5
    output_frames=20
    evaluator = SimplifiedEvaluator(
        seq_len=20,
        value_scale=90,
        thresholds=[20, 30, 35, 40]
    )
    frame_interval=6

print("-----准备模型-----")
model, optimizer, criterion, lr_scheduler, early_stop, ex_lr_scheduler, norm_clip, if_save_epoch, base_LR, early_stop = create_models(model_name,input_frames,output_frames,frame_interval,dataset)

if torch.cuda.device_count() > 1:
    model = SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.DataParallel(model)
    print("use gpus:", torch.cuda.device_count())

model.to(device)
print(model_name)

loss_fn = criterion
save_path = os.path.join(config_root, 'model_params',dataset)
model_folder = os.path.join(save_path, model_name)
file_name = f"{model_name}_best.pth"
file_name = f"{model_name}_best 82835591.pth"
save_best_path = os.path.join(model_folder, file_name)
if os.path.exists(save_best_path):
    model.load_state_dict(torch.load(save_best_path))
    print(f"成功加载模型参数：{save_best_path}")
else:
    print(f"未找到模型参数文件：{save_best_path}")

print('-----------------test best-----------------')
model.eval()
gts=[]
preds=[]
total_test_loss = 0
with torch.no_grad():
    for data in test_dataloader:
        imgs, targets = data[:,:input_frames,:,:,:],data[:,input_frames:,:,:,:]
        imgs_cuda, targets_cuda = imgs.to(torch.float32).to(device), targets.to(torch.float32).to(device)
        outputs = model(imgs_cuda)
        loss = loss_fn(outputs, targets_cuda)
        total_test_loss = total_test_loss + loss
        outputs=outputs.cpu().numpy()
        targets_cuda=targets_cuda.cpu().numpy()
        gts.append(targets_cuda.squeeze(2))
        preds.append(outputs.squeeze(2))
print("最优参数测试集上的Loss: {}".format(total_test_loss))
if dataset == 'LAPS':
    preds = np.array(preds).reshape(-1)
    gts = np.array(gts).reshape(-1)
    SimplifiedEvaluator(preds,gts,thresholds)
elif dataset == 'Shanghai':
    preds = np.array(preds)
    preds = preds.reshape(-1, preds.shape[2], preds.shape[3], preds.shape[4]).astype(np.float32)
    gts = np.array(gts)
    gts = gts.reshape(-1, gts.shape[2], gts.shape[3], gts.shape[4]).astype(np.float32)
    evaluator.evaluate(preds, gts)
    results = evaluator.done()
    for thresh, metrics in results["threshold_metrics"].items():
        print(f"{thresh}mm CSI: {metrics['CSI']:.4f}  HSS: {metrics['HSS']:.4f}")
    print("\nOverall Metrics:")
    print(f"FAR:  {results['FAR']:.4f}")
    print(f"RMSE: {results['RMSE']:.2f}")
    print(f"SSIM: {results['SSIM']:.4f}")
    print(f"LPIPS: {results['LPIPS']:.4f}")

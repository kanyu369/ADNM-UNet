# python -m train
import torch 
import torchvision
from train_untils import create_models
from torch.utils.data import DataLoader
import numpy as np
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

# model_name='ADNMUnet'
# model_name='TrajGRU'
# model_name='ConvLSTM'
# model_name='TransUnet'
# model_name='LPTQPN'
# model_name='SmaATUnet'
model_name='SwinUnet'

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
    
save_path = os.path.join(config_root, 'model_params')
save_path = os.path.join(save_path, dataset)
os.makedirs(save_path, exist_ok=True)

save_result_path = os.path.join(config_root, 'results')
os.makedirs(save_result_path, exist_ok=True)

print("-----准备模型-----")
model, optimizer, criterion, lr_scheduler, if_early_stop, ex_lr_scheduler, norm_clip, if_save_epoch, base_LR, early_stop = create_models(model_name,input_frames,output_frames,frame_interval,dataset)

if model_name == 'ADNMUnet':
    warmup_epoch=3
    epochs = 40
    if frame_interval < 120/input_frames: # 根据InstanceNorm决定梯度裁剪强度
        save_epoch=34
        norm_ratio=1.75
        norm_max=0.025
        norm_initial=0.175
        early_stop=3
        grad_epoch_excursion=1
    else:
        save_epoch=20
        norm_ratio=3.0
        norm_max=0.035
        norm_initial=0.065
        early_stop=5
        grad_epoch_excursion=0
        
if if_early_stop:
    early_stop_count=0

if torch.cuda.device_count() > 1:
    model = SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.DataParallel(model)
    print("use gpus:", torch.cuda.device_count())

model.to(device)
print(model_name)

total_train_step = 0


loss_fn = criterion

prev_avg_grad_norm = 0.0
for epoch in range(1,epochs+1):
    total_train_step += 1
    print("-----第{}轮训练开始-----".format(epoch))
    test_loss=0
    gts=[]
    preds=[]
    total_grad_norm = 0.0
    clip_count = 0

    if norm_clip:
        if epoch <= warmup_epoch+1:
            current_norm = norm_max
        elif epoch <= save_epoch- warmup_epoch+grad_epoch_excursion:

            alpha =norm_initial+(1-norm_initial)*(epoch - warmup_epoch) / (save_epoch- warmup_epoch+grad_epoch_excursion) 
            current_norm = alpha*norm_ratio * prev_avg_grad_norm
        else:
            current_norm = norm_ratio * prev_avg_grad_norm
        
    for data in train_dataloader:
        imgs, targets = data[:,:input_frames,:,:,:],data[:,input_frames:,:,:,:]
        imgs_cuda, targets_cuda = imgs.to(device), targets.to(device)
        model.train()
        outputs = model(imgs_cuda)
        loss = loss_fn(outputs, targets_cuda)
        loss.backward()
        if norm_clip:
            original_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), current_norm)
            total_grad_norm += original_norm.item()
            if original_norm > current_norm:
                clip_count += 1
        optimizer.step()
        optimizer.zero_grad()
        test_loss += loss.item()
    print("训练集损失：",test_loss)

    if norm_clip:
        avg_grad_norm = total_grad_norm / len(train_dataloader)
        prev_avg_grad_norm = avg_grad_norm
        clip_ratio = clip_count / len(train_dataloader)
        print(f"梯度阈值:{current_norm:.4f}, 平均梯度范数: {avg_grad_norm:.4f}, 裁剪率: {clip_ratio:.4f}")

    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            imgs, targets = data[:,:input_frames,:,:,:],data[:,input_frames:,:,:,:]
            imgs_cuda, targets_cuda = imgs.to(torch.float32).to(device), targets.to(torch.float32).to(device)
            outputs = model(imgs_cuda)
            val_loss = val_loss + loss_fn(outputs, targets_cuda).item()
            outputs=outputs.cpu().numpy()
            targets_cuda=targets_cuda.cpu().numpy()
            gts.append(targets_cuda.squeeze(2))
            preds.append(outputs.squeeze(2))

    if (if_save_epoch is False and val_loss<best) or (if_save_epoch is True and epoch>save_epoch and val_loss<best):
        model_folder = os.path.join(save_path, model_name)
        os.makedirs(model_folder, exist_ok=True)
        file_name = f"{model_name}_best.pth"
        save_best_path = os.path.join(model_folder, file_name)
        torch.save(model.state_dict(), save_best_path)
        print("best_val_loss", val_loss)
        if (if_early_stop is True and if_save_epoch is False and val_loss<best) or (if_early_stop is True and if_save_epoch is True and epoch>save_epoch and val_loss<best):
            early_stop_count=0
        best = val_loss
    else:
        print("val_loss", val_loss)
        if (if_early_stop is True and if_save_epoch is False) or (if_early_stop is True and if_save_epoch is True and epoch>save_epoch):
            early_stop_count+=1

    
    # 更新学习率调度器
    if ex_lr_scheduler:
        lr_scheduler.step()
    else:
        lr = base_LR * (1.0 - total_train_step / epochs) ** 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data[:,:input_frames,:,:,:],data[:,input_frames:,:,:,:]
            imgs_cuda, targets_cuda = imgs.to(torch.float32).to(device), targets.to(torch.float32).to(device)
            outputs = model(imgs_cuda)
            loss = loss_fn(outputs, targets_cuda)
            total_test_loss = total_test_loss + loss
    print("整体测试集上的Loss: {}".format(total_test_loss))


    if if_early_stop and early_stop_count >= early_stop:
        break
        

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
        # print(f"{thresh}mm CSI: {metrics['CSI']:.4f} POD: {metrics['POD']:.4f} HSS: {metrics['HSS']:.4f}")
        print(f"{thresh}mm CSI: {metrics['CSI']:.4f} HSS: {metrics['HSS']:.4f}")
    print("\nOverall Metrics:")
    print(f"FAR:  {results['FAR']:.4f}")
    print(f"RMSE: {results['RMSE']:.2f}")
    print(f"SSIM: {results['SSIM']:.4f}")
    print(f"LPIPS: {results['LPIPS']:.4f}")
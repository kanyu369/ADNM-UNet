import numpy as np
import logging
import lpips
import torch
import cv2

np.seterr(divide='ignore', invalid='ignore')


def print_log(message):
    print(message)
    logging.info(message)

class SimplifiedEvaluator:
    def __init__(self, seq_len, value_scale, thresholds=[20, 30, 35, 40]):
        self.metrics = {}
        self.thresholds = thresholds
        for threshold in self.thresholds:
            self.metrics[threshold] = {
                "hits": [],
                "misses": [],
                "falsealarms": [],
                "correctnegs": [],
            }
        self.losses = {
            "mse":  [],
            "mae":  [],
            "rmse": [],
            "psnr": [],
            "ssim": [],
            "lpips": [],
        }
        self.seq_len = seq_len
        self.total = 0
        self.value_scale = value_scale

        self.lpips_fn = lpips.LPIPS(net='alex', verbose=False)
        if torch.cuda.is_available():
            self.lpips_fn.cuda()
        self.TP=[]
        self.TN=[]
        self.FP=[]
        self.FN=[]

    def float2int(self, arr):
        x = arr.clip(0.0, 1.0)
        return (x * self.value_scale).astype(np.uint16)

    def evaluate(self, true_batch, pred_batch):
        if isinstance(pred_batch, torch.Tensor):
            pred_batch = pred_batch.detach().cpu().numpy()
            true_batch = true_batch.detach().cpu().numpy()
        
        pred_batch = pred_batch.clip(0.0, 1.0)
        true_batch = true_batch.clip(0.0, 1.0)

        batch_size, seq_len = true_batch.shape[:2]
        
        # 计算LPIPS
        lpips_batch = self._cal_batch_lpips(pred_batch, true_batch)
        self.losses['lpips'].extend(lpips_batch)
        
        # 转换到整数范围
        pred = self.float2int(pred_batch)
        gt = self.float2int(true_batch)
        
        # 计算阈值相关指标
        for threshold in self.thresholds:
            for b in range(batch_size):
                seq_hit, seq_miss, seq_fa, seq_cn = [], [], [], []
                for t in range(seq_len):   
                    hit, miss, fa, cn = self._cal_frame(gt[b][t], pred[b][t], threshold)
                    seq_hit.append(hit)
                    seq_miss.append(miss)
                    seq_fa.append(fa)
                    seq_cn.append(cn)
                
                self.metrics[threshold]["hits"].append(seq_hit)
                self.metrics[threshold]["misses"].append(seq_miss)
                self.metrics[threshold]["falsealarms"].append(seq_fa)
                self.metrics[threshold]["correctnegs"].append(seq_cn)

        # 计算回归指标
        for b in range(batch_size):
            seq_mse, seq_mae, seq_rmse, seq_psnr, seq_ssim = [], [], [], [], []
            for t in range(seq_len):
                mae, mse, rmse, psnr, ssim = self._cal_frame_losses(
                    true_batch[b][t], 
                    pred_batch[b][t]
                )
                seq_mse.append(mse)
                seq_mae.append(mae)
                seq_rmse.append(rmse)
                seq_psnr.append(psnr)
                seq_ssim.append(ssim)
            
            self.losses['mse'].append(seq_mse)
            self.losses['mae'].append(seq_mae)
            self.losses['rmse'].append(seq_rmse)
            self.losses['psnr'].append(seq_psnr)
            self.losses['ssim'].append(seq_ssim)
        
        self.total += batch_size

    def _cal_frame(self, obs, sim, threshold):
        obs_bin = (obs >= threshold).astype(int)
        sim_bin = (sim >= threshold).astype(int)
        
        TP = np.sum((obs_bin == 1) & (sim_bin == 1))
        FN = np.sum((obs_bin == 1) & (sim_bin == 0))
        FP = np.sum((obs_bin == 0) & (sim_bin == 1))
        TN = np.sum((obs_bin == 0) & (sim_bin == 0))
        
        return TP, FN, FP, TN

    def _cal_frame_losses(self, true, pred):
        pred = pred * self.value_scale
        true = true * self.value_scale
        
        mae = np.mean(np.abs(pred - true))
        mse = np.mean((pred - true) ** 2)
        rmse = np.sqrt(mse)
        psnr = 20 * np.log10(self.value_scale / np.sqrt(mse))
        
        # 计算SSIM
        pred = pred.astype(np.float32)
        true = true.astype(np.float32)
        ssim = self.cal_ssim(pred, true)
        
        return mae, mse, rmse, psnr, ssim

    def cal_ssim(self, pred, true):
        C1 = (0.01 * self.value_scale)**2
        C2 = (0.03 * self.value_scale)**2
    
        img1 = pred.astype(np.float64)
        img2 = true.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())
    
        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    def _cal_batch_lpips(self, preds, trues):
        def _to_tensor(arr):
            t = torch.from_numpy(arr).float()
            t = t.unsqueeze(2).repeat(1, 1, 3, 1, 1)  # Add RGB channels
            t = t * 2 - 1  # LPIPS需要[-1,1]范围
            return t.cuda() if torch.cuda.is_available() else t

        preds = _to_tensor(preds)
        trues = _to_tensor(trues)
        
        lpips_vals = []
        for t in range(preds.shape[1]):
            lpips_val = self.lpips_fn(preds[:, t], trues[:, t])
            lpips_vals.append(lpips_val.detach().cpu().numpy())
        
        return np.array(lpips_vals).squeeze().T.tolist()

    # def done(self):
    #     results = {}
        
    #     # 处理阈值指标
    #     threshold_metrics = {}
    #     all_far = []
        
    #     for threshold in self.thresholds:
    #         hits = np.array(self.metrics[threshold]["hits"])
    #         misses = np.array(self.metrics[threshold]["misses"])
    #         falsealarms = np.array(self.metrics[threshold]["falsealarms"])
            
    #         # 转换为numpy数组并处理NaN
    #         hits = np.nan_to_num(hits)
    #         misses = np.nan_to_num(misses)
    #         falsealarms = np.nan_to_num(falsealarms)
            
    #         # 计算各指标
    #         CSI = np.mean(hits) / (np.mean(hits) + np.mean(misses) + np.mean(falsealarms))
    #         POD = np.mean(hits) / (np.mean(hits) + np.mean(misses))
    #         HSS = (2*(np.mean(hits)*np.mean(self.metrics[threshold]["correctnegs"]) - 
    #                  np.mean(misses)*np.mean(falsealarms))) / (
    #             (np.mean(hits)+np.mean(misses))*(np.mean(misses)+np.mean(self.metrics[threshold]["correctnegs"])) + 
    #             (np.mean(hits)+np.mean(falsealarms))*(np.mean(falsealarms)+np.mean(self.metrics[threshold]["correctnegs"])))
            
    #         FAR = np.mean(falsealarms) / (np.mean(hits) + np.mean(falsealarms))
    #         all_far.append(FAR)
            
    #         threshold_metrics[threshold] = {
    #             "CSI": CSI,
    #             "POD": POD,
    #             "HSS": HSS
    #         }
        
    #     # 处理回归指标
    #     rmse = np.mean(np.sqrt(np.mean(self.losses['mse'], axis=0)))
    #     ssim = np.mean(self.losses['ssim'])
    #     lpips = np.mean(self.losses['lpips'])
        
    #     return {
    #         "threshold_metrics": threshold_metrics,
    #         "FAR": np.mean(all_far),
    #         "RMSE": rmse,
    #         "SSIM": ssim,
    #         "LPIPS": lpips
    #     }

    def done(self):
        results = {}

        TP_sum=0
        TN_sum=0
        FP_sum=0
        FN_sum=0

        threshold_metrics = {}
        all_far = []
        
        for threshold in self.thresholds:
            hits = np.array(self.metrics[threshold]["hits"])
            misses = np.array(self.metrics[threshold]["misses"])
            falsealarms = np.array(self.metrics[threshold]["falsealarms"])
            correctnegs = np.array(self.metrics[threshold]["correctnegs"])
            
            hits = np.nan_to_num(hits)
            misses = np.nan_to_num(misses)
            falsealarms = np.nan_to_num(falsealarms)
            correctnegs = np.nan_to_num(correctnegs)
            
            TP = np.sum(hits)
            TN = np.sum(correctnegs)
            FP = np.sum(falsealarms)
            FN = np.sum(misses)
            
            # print(f"Threshold {threshold}:")
            # print(f"  TP: {TP}")
            # print(f"  TN: {TN}")
            # print(f"  FP: {FP}")
            # print(f"  FN: {FN}")

            TP_sum += TP
            TN_sum += TN
            FP_sum += FP
            FN_sum += FN
            
            CSI = TP / (TP + FP + FN)
            POD = TP / (TP + FN)
            
            HSS = (2 * (TP * TN - FP * FN)) / (
                (FP**2 + FN**2 + 2 * TP * TN + (FP + FN) * (TP + TN))
            )
            
            FAR = FP / (TP + FP)
            all_far.append(FAR)
            
            threshold_metrics[threshold] = {
                "TP":TP,
                "TN":TN,
                "FP":FP,
                "FN":FN,
                "CSI": CSI,
                "POD": POD,
                "HSS": HSS
            }
        self.TP.append(TP_sum/len(self.thresholds))
        self.FP.append(FP_sum/len(self.thresholds))
        self.TN.append(TN_sum/len(self.thresholds))
        self.FN.append(FN_sum/len(self.thresholds))
        
        rmse = np.mean(np.sqrt(np.mean(self.losses['mse'], axis=0)))
        ssim = np.mean(self.losses['ssim'])
        lpips = np.mean(self.losses['lpips'])
        
        return {
            "threshold_metrics": threshold_metrics,
            "FAR": np.mean(all_far),
            "RMSE": rmse,
            "SSIM": ssim,
            "LPIPS": lpips
        }

    def reset(self):
        """重置所有评估指标和计数器"""
        # 重置阈值相关指标
        for threshold in self.thresholds:
            self.metrics[threshold] = {
                "hits": [],
                "misses": [],
                "falsealarms": [],
                "correctnegs": [],
            }
        
        # 重置回归指标
        self.losses = {
            "mse":  [],
            "mae":  [],
            "rmse": [],
            "psnr": [],
            "ssim": [],
            "lpips": [],
        }
        
        # 重置样本计数器
        self.total = 0

# 使用示例
if __name__ == "__main__":
    # 初始化评估器
    evaluator = SimplifiedEvaluator(
        seq_len=20,
        value_scale=90,
        thresholds=[20, 30, 35, 40]
    )
    
    # 模拟数据（batch_size=2, seq_len=6, 64x64）
    pred = np.random.rand(2, 6, 64, 64).clip(0, 1)
    true = np.random.rand(2, 6, 64, 64).clip(0, 1)
    
    # 执行评估
    evaluator.evaluate(true, pred)
    
    # 获取结果
    results = evaluator.done()
    
    # 打印结果
    print("\nThreshold Metrics:")
    for thresh, metrics in results["threshold_metrics"].items():
        print(f"{thresh}mm:")
        print(f"  CSI: {metrics['CSI']:.4f}")
        print(f"  POD: {metrics['POD']:.4f}")
        print(f"  HSS: {metrics['HSS']:.4f}")
    
    print("\nOverall Metrics:")
    print(f"FAR:  {results['FAR']:.4f}")
    print(f"RMSE: {results['RMSE']:.2f}")
    print(f"SSIM: {results['SSIM']:.4f}")
    print(f"LPIPS: {results['LPIPS']:.4f}")
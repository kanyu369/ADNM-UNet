# ADNM-UNet

### 📋 Project Overview
In this project, we provide:
* **Complete ADNM-UNet Code**: Full implementation of our proposed architecture.
* **Baseline Models**: A collection of benchmark models for comparative analysis.
* **Full Pipeline**: Integrated scripts for both **Training** and **Inference** (Testing).

![ADNM-UNet](https://github.com/kanyu369/ADNM-UNet/blob/main/ADNMUnet.png)

1\. Although we have provided `requirements.txt`, we still recommend manually installing the `mamba_ssm` environment in [Mamba2](https://github.com/state-spaces/mamba) to avoid environmental errors.

If you only need to run **ADNM-UNet** and the provided **baseline** models without installing `mamba_ssm`, follow these steps:

   - **Step 1: Refactor `RMSNorm`**
     Implement a standalone `RMSNorm` (Root Mean Square Layer Normalization) to replace the dependency. You can use the following PyTorch implementation:

     ```python
     import torch
     import torch.nn as nn

     class RMSNorm(nn.Module):
         def __init__(self, d_model: int, eps: float = 1e-5):
             super().__init__()
             self.eps = eps
             self.weight = nn.Parameter(torch.ones(d_model))

         def forward(self, x):
             output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
             return output * self.weight.to(x.dtype)
     ```

   - **Step 2: Modify Model Files**
     - **Update Imports**: In `models/ADNM-UNet`, redirect the `RMSNorm` import to your local implementation defined above.
     - **Comment Out Dependencies**: Locate and comment out all `import mamba_ssm` or `from mamba_ssm import ...` statements in:
       - `models/ADN-SSD`
       - `models/ADNM-UNet`
<br>
<br>
2\. In `config.py`, modify `config_root` to the project's root directory.
<br>
<br>
3\. To obtain the `shanghai.h5` dataset, please visit [DiffCast](https://github.com/DeminYu98/DiffCast).
<br>
<br>
4\. To train models, cd to the project's root directory and run
```python
python -m train
```

5\. To validate models after training, cd to the project's root directory and run
```python
python -m validate
```

6\. To generate pictures, cd to the project's root directory and run
```python
python -m pic_results
```


This project is based on VSSD([paper](https://arxiv.org/abs/2407.18559),[code](https://github.com/YuHengsss/VSSD)), Mamba-UNet([paper](https://ieeexplore.ieee.org/abstract/document/10925469/)), Mamba2 ([paper](https://arxiv.org/abs/2405.21060), [code](https://github.com/state-spaces/mamba)), DiffCast([paper](https://arxiv.org/abs/2312.06734), [code](https://github.com/DeminYu98/DiffCast)), thanks for their excellent works.

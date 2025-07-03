# ADNM-UNet
![ADNM-UNet](https://github.com/kanyu369/ADNM-UNet/blob/main/ADNMUnet.png)

1\. Although we have provided `requirements.txt`, we still recommend manually installing the mamba environment in [Mamba2](https://github.com/state-spaces/mamba) to avoid environmental errors.
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

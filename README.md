# ADNM-UNet

1\. Although we have provided requirements.txt, we recommend manually installing the mamba environment in [mamba](https://github.com/state-spaces/mamba)
2\. In `config.py`, modify `config_root` to the project's root directory.
3\. To obtain the `shanghai.h5` dataset, please visit [DiffCast](https://github.com/DeminYu98/DiffCast).
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

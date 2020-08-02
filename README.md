# 3AUNet
Pytorch implementation of retinal vessel segmentation using 3AUNet

### Motivation ###
Vessel segmentation in retina is an indispensable part of the task of the automatic detection of retinopathy through fundus images, while there are several challenges, such as lots of image noise, low distinction between blood vessels and environment, and uneven distribution of thick and thin blood vessels.

### contact me ###
If you have any questions, you can leave a message in the issue or send it to me by email.  My email:
```bash
xxxx@163.com
```

### License me ###
This code is available only for non-commercial use.

### How to Start ###
Setting：
Constants.py
```bash
ROOT = './dataset/CHASEDB'
or
ROOT = './dataset/DRIVE'
```
Training:
```bash
python train.py
```
Testing:
```bash
python test_DRIVE.py
or
python test_CHASEDB.py
```

### Thanks ###
Some implementation code comes from：

CE-Net：https://github.com/Guzaiwang/CE-Net

DANet：https://github.com/junfu1115/DANet

ECANet：https://github.com/BangguWu/ECANet



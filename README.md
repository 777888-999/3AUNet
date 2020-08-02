# 3AUNet
Pytorch implementation of retinal vessel segmentation using 3AUNet


### Training details ###
To train the model, simply run this command:
```bash
python train.py --save-path /path/to/model.pt 
```
On a recent GPU, it takes 30 min per epoch, so ~12h for 25 epochs. 
You should get a model that scores `0.71 +/- 0.01` in `MMA@3` on HPatches (this standard-deviation is similar to what is reported in Table 1 of the paper). 

Note that you can fully configure the training (i.e. select the data sources, change the batch size, learning rate, number of epochs etc.). One easy way to improve the model is to train for more epochs, e.g. `--epochs 50`. For more details about all parameters, run `python train.py --help`.

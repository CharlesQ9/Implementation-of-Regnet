# Implementation-of-Regnet
Paper: [RegNet: Multimodal Sensor Registration Using Deep Neural Networks](https://arxiv.org/pdf/1707.03167.pdf).

This implementation is based on https://github.com/aaronlws95/regnet

Thanks for the author Nick Schneider providing some details of the model.

## Training
The original implementation by aaronlws95(https://github.com/aaronlws95) 
```
python train.py
```

Our implementation for RegNet with original parameters:
```
python trainv3.py
```
Our implementation for RegNet with learnable loss:
```
python trainv3learn.py
```
Our implementation for Regnet with simple fusion module:
```
python train_fusion.py
```
Our implementation for Regnet with simple fusion module and learnable loss:
```
train_fusionlearn.py
```

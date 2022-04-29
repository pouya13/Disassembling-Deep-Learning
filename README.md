# Deep Learning based Disassembler
This repository contains codes for a deep learning based disassembler using power side-channel traces gathered from AVR atmega8 microprocessor. The data gathering mechanism, labeling and also models are described below.

# Dataset
The dataset is obtained using a custom designed Printed Circuit Board (PCB) using Altium Designer. The printed circuit board contains an Arduino nano, which is used as ISP programmer. The target processor, i.e. atmega8, is working with 8MHz external oscillator. To generate labeled dataset and record side-channel traces, we have used [ARCSim](https://github.com/pouya13/ARCSim)  (For more information please refer to the reference paper and the corresponding page for ARCSim).

# Models
We have used two different modelsو which are customized versions of [VGG](https://arxiv.org/abs/1409.1556) and [ResNet18](https://arxiv.org/abs/1512.03385). Furthermore, we have used RNN and LSTM networks for instruction probability prediction. More details are available at our paper.

# Instruction Probability Prediction
Instructions in real programs do not follow each other, randomly. Some instructions never come after each other, or some instructions always follow each other. Hence, we have proposed a RNN based structure to model the probability distributions and estimate the probability of each instruction by knowing some previous ones.

# How to Run
By running the train.py file, the data will be loaded and the model training will begin.

# Contact Information
For more information and comments, you can send emails to pouyanarimani.kh@gmail.com.

# Citation

```
```



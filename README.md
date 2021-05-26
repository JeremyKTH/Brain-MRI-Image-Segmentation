# Brain MRI Image Segmentation with U-Net
The network architecture in this project is based on the original U-Net publication [U-Net: ConvolutionalNetworks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf). The codes in this repo are mainly developed in Python and GoogleColab.

## Table of contents

<!--ts-->
   * [Introduction](#Introduction)
   * [Requirements](#Requirements)
   * [Experiments](#Experiments)
   * [Results](#Results)
   * [Acknowledgements](#Acknowledgements)
<!--te-->


## Introduction
The U-Net architecture of this project consists of an encoder and a decoder part that together givethe network an U-shaped form. The encoder part follows a traditional architecture of aconvolutional network. Each layer consists of a repeated 3x3 padded double convolutions, followed by batch normalization and a rectifier linear units (ReLU) activation.  The output feature map is stored as a skip connection which will later be concatenated to the decoder part. Moving downwardsthrough the network, a down sampling is performed by a 2x2 max pooling operation with a stride of2, which doubles the number of feature channels.

![UNet](https://github.com/JeremyKTH/Brain-MRI-Image-Segmentation/blob/main/UNET-architecture%20.jpg)

## Dataset
[Brain MRI images](https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation)

## Requirements
- Python  3.9.1
- PyTorch 1.8.1
- CUDA toolkit 10.2
- Albumentations

## Experiments
1. Optimiser Comparison - SGD vs Adam
2. Loss Comparison - Binary Cross-Entropy(BCE) Loss vs Dice Loss vs BCE + Dice


## Results
In this project, it was found that Adam optimiser with Binary Cross-Entropy Loss produce the best test result.

### SGD + BCE (50 epochs)
![50_sgd_bce](https://github.com/JeremyKTH/Brain-MRI-Image-Segmentation/blob/main/Predictions/ForREADME/50_bce_sgd.PNG)
### Adam + BCE (50 epochs)
![50_adam_bce](https://github.com/JeremyKTH/Brain-MRI-Image-Segmentation/blob/main/Predictions/ForREADME/50_adam_bce.PNG)
### Adam + Dice (50 epochs)
![50_adam_dice](https://github.com/JeremyKTH/Brain-MRI-Image-Segmentation/blob/main/Predictions/ForREADME/50_adam_dice.PNG)
### Adam + BCE + BCE (50 epochs)
![50_adam_bce_dice](https://github.com/JeremyKTH/Brain-MRI-Image-Segmentation/blob/main/Predictions/ForREADME/50_adam_bce_dice.PNG)

<!-- CONTACT -->
## Contact
- Chieh-Ju Wu (Jeremy) - jeremy.cjwukth@gmail.com
- Fredrik Mazur - fredrik@mazur.se
- Niclas Määttä - niclas.maatta@hotmail.com
- Daniel Grönås - daniel.gronas@hotmail.com

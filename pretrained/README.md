## Folder Structure

### 1. CNN

This folder contains the Convolutional Neural Network (CNN) models:

- `resnet18.pth`: The original ResNet18 model.
- `resnet18-sam.pth`: Add spatial attention mechanism to ResNet18 model.

### 2. GAN

This folder contains the Generative Adversarial Network (GAN) models:

- `sh-dcgan.pth`: Add Spectral Normalization to DCGAN model. In training, the loss function adopts the hinge adversarial
  loss.

### 3. Modified ResNet18

This model pretrained on ImageNet with a modified fully connected (fc) layer for 2 outputs:

- `new_resnet18.pth`: The modified ResNet18 model.
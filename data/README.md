# Data🖼️

This folder contains two types of data for training a CNN classification model: 

1、The original `pneumoniamnist` dataset， "real".

2、The synthetic data generated using DCGAN, "fake".

## Pneumoniamnist Dataset🩺

The `pneumoniamnist` dataset consists of X-ray images of lungs that have been labeled as either normal or containing pneumonia.

----

Basic information：

| Feature |      Description      |
| :-----: | :-------------------: |
|  Name   |   `pneumoniamnist`    |
|  Type   | Image dataset（.png） |
|  Size   |     5,856 images      |
| Classes |   Normal, Pneumonia   |
| Source  | https://medmnist.com/ |

Furthermore, it includes the number and proportion of ==positive and negative== samples in the data set：

|       | Numbers | Normal | Pneumonia |  Ratio  |
| :---: | :-----: | :----: | :-------: | :-----: |
| Train |  4708   |  1214  |   3494    | 1：2.87 |
|  Val  |   524   |  135   |    389    | 1：2.88 |
| Test  |   624   |  234   |    390    | 1：1.67 |
| TOTAL |  5856   |  1583  |   4273    | 1：2.69 |

Note：

* To use this dataset, simply load the images from their respective subfolders within the `data` directory and specify the corresponding labels during training.

* Here I provide the loading method of the dataset, refer to `utils/classifier_dataset.py` .

## Synthetic Data🤖

In addition to the `pneumoniamnist` dataset, we also provide synthetic data generated using DCGAN. This data is intended to balance out the ratio of positive and negative samples during training.

Both datasets are stored in subfolders within this directory. To use them, you can modify your training code to load the data from these folders.

## Fusion of real and fake datasets🌀

The synthetic fake data and the original real data are fused to obtain the data set used for training, and the specific distribution is as follows:

|       | Numbers | Normal | Pneumonia |
| ----- | ------- | ------ | --------- |
| Train | 6988    | 3494   | 3494      |
| Val   | 778     | 389    | 389       |
| Test  | 752     | 376    | 376       |

Quantitative comparison before and after data augmentation:

| Class     | Before augmentation | After augmentation |
| --------- | ------------------- | ------------------ |
| Normal    | 1583                | 4259               |
| Pneumonia | 4273                | 4259               |

That's all!!😊
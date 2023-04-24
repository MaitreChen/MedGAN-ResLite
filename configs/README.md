# Config 🛠️

This folder contains the `config.yaml` file, which specifies the configuration settings for training a CNN classification model.

## Usage 🔧

The settings in `config.yaml` are designed to work with 2D medical image datasets in the MedMNIST collection. While this project specifically uses the `pneumoniamnist` dataset, you can modify the corresponding fields in `config.yaml` to match the characteristics of your specific dataset.

---

Of course, this project isn't just for the pneumoniamnist dataset, you can use any 2D dataset in MedMNIST！

For example, if you are using the `breastmnist` dataset, which consists of 3-channel images, you would need to change the `n_channels` field to 3.

## YAML Contents 📝

The `config.yaml` file includes the following fields:

```
Copy Code# data setting
data_name: pneumonia
n_channels: 1
n_classes: 2
classes: ('normal', 'pneumonia')

# root of real dataset
root1: './data/pneumoniamnist'

#root of fake dataset
root2: './data/fake_new'

# model input size
image_size: 224
```

These settings describe the name of the dataset (`data_name`) 🩺, number of channels in the input images (`n_channels`) 🔍, number of classes (`n_classes`) 🔢, class labels (`classes`)  and paths to the real and synthetic datasets (`root1` and `root2`, respectively) , and the input image size (`image_size`) .

----

That's all! 😊
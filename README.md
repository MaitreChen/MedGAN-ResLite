<div align="center">
  <a href="" target="_blank">
  <img width="100%" src="https://github.com/MaitreChen/MedGAN-ResLite/blob/main/figures/logo.png"></a>
</div>

<div align="center">
   <a href="https://img.shields.io/badge/Nickname-阿斌~-blue"><img src="https://img.shields.io/badge/Nickname-阿斌~-blue.svg"></a>
   <a href="https://img.shields.io/badge/Hello-Buddy~-red"><img src="https://img.shields.io/badge/Hello-Buddy~-red.svg"></a>
   <a href="https://img.shields.io/badge/Enjoy-Yourself-brightgreen"><img src="https://img.shields.io/badge/Enjoy-Yourself-brightgreen.svg"></a>
</div>


# 📣Introduction

This is a **pneumonia classification** project that addresses the issue of **class imbalance** by utilizing generative adversarial networks (GAN) to generate images of minority class samples. In addition, the **spatial attention mechanism** is introduced into **ResNet18** to enhance the generalization performance of classifier!

🔥 **Workflow**

<div align="center">
  <a href="" target="_blank">
  <img width="100%" src="https://github.com/MaitreChen/MedGAN-ResLite/blob/main/figures/workflow.png"></a>
</div>



# 🚩Updates & Roadmap

###  🌟**New Updates**

* ✅ Mar 21, 2023. Creat "MedGAN-ResLite" project repository and Find MedMNIST.
* ✅ Mar 22, 2023. Generate pneumonia samples with DCGAN.
* ✅ May 30, 2023. Replace original Loss function with Hinge Adversial Loss.
* ✅ Apri 1, 2023.  DCGAN + Spectral Normalization.
* ✅ Apri 4, 2023.  Add DCGAN metrics：Inception Score + FID + KID; Fuse and Split dataset;
* ✅ Apri 5, 2023.  Override the dataset inheritance class.
* ✅ Apri 6, 2023.  Write train, eval and infer scripts for classifier. And get a new-model by modifing input & output shape of pre-trained model. Add metrics：acc + auc + f1 + confusion matrix.
* ✅ April 7, 2023. Add scripts: export_onnx.py  and inference_onnx.py.
* ✅ April 8, 2023. Tuning the hyperparameters of DCGAN.
* ✅ April 10, 2023.  Explore CBAM attention mechanism to add location and quantity.
* ✅ April 14, 2023. Abalation Study: GAN, DCGAN, DCGAN+Hinge, DCGAN + SN, DCGAN + SH.
* ✅ April 21, 2023. Attention mechanism visualization using CAM.
* ✅ April 25, 2023. Make a Presentation. 



----



### 💤Progress & Upcoming work

| ✅                           | ✅❗                    | ❓            |
| --------------------------- | --------------------- | ------------ |
| Finished, and Successfully! | Finished, but Failed! | Unfinished！ |

#### Part 1: Dataset and Preprocessing

* ❓ Experiment with more challenging datasets, such as [ChestXRay2017](https://data.mendeley.com/datasets/rscbjbr9sj/2), Kaggle, etc.
* ❓ Consider introducing the idea of **"learning"** when scaling the image, such as adopting **transposed convolution** instead of interpolation when scaling up the image size

#### Part 2: Generation part

* ✅❗ May 23, 2023. Try Muti-Scale Fusion.
* ✅❗ May 25, 2023. Introduce class information into DCGAN to generate samples.【cDCGAN】
* ❓ Replace original Loss function with Wasserstein distance.

#### Part 3: Classification part

* ❓ Apply ensemble learning methods, such as voting evaluation.

#### Part 4: Lightweight-NN part

* ❓ Pruning：one-hot + iterative ，including L1✅、L2✅、FPGM✅、BNScale.
* ❓ Build the pruned model automatically.
* ❓ Knowledge distillation：design lightweight network A，and use pruned-model to guide A.

#### Part 5: Depolyment part

* ❓ Deploy model on CPU and NSC2 using OpenVINO. 【Python ✅and C++ version】.
* ❓ Deploy on the web side using Django or flask.

#### Other

* ❓ Explore the influence of **attention mechanism** on deep network and shallow network.



# ✨Usage

## Install

Clone repo and install [requirements.txt](https://github.com/MaitreChen/MedGAN-ResLite/blob/main/requirements.txt).

```bash
git clone git@github.com:MaitreChen/MedGAN-ResLite.git
cd MedGAN-ResLite
pip install -r requirements.txt
```

## Preparations

You can download dataset from this [link](https://drive.google.com/drive/folders/1DOG37hN4LZ8rwj1Mfxlt9qcZwaQetX2S?usp=share_link). It includes the pneumoniamnist original real dataset and the fake dataset synthesized using DCGAN (see data [README.md](https://github.com/MaitreChen/MedGAN-ResLite/blob/main/data/README.md) for details)

The dataset **structure** directory is as follows:

```bash
MedGAN-ResLite/
|__ data/
    |__ real/
        |__ train/
            |__ normal/
                |__ img_1.png
                |__ ...
            |__ pneumonia/
                |__ img_1.png
                |__ ...
        |__ val/
            |__ normal/
            |__ pneumonia/
        |__ test/
            |__ ...
    |__ fake/
        |__ ...
```

You can download pretrained checkpoints from this [link](https://drive.google.com/drive/folders/1iMSjrbF0zWCtYfCxdyY_yOhMbn9KUZM-?usp=share_link) and put it in your **pretrained/** folder. It contains **resnet18-sam** and **sh-dcgan** model.

## Inference

### Classification part

🚀Quick start, and the results will be saved in the **figures/classifier_torch** folder.

```bash
python infer_classifier.py --ckpt-path pretrained/resnet18-sam.pth --image-path imgs/pneumonia_img1.png
```

🌜Here are the options in more detail:

| Option       | Description                                                  |
| ------------ | ------------------------------------------------------------ |
| --ckpt-path  | Checkpoints path to load the pre-trained weights for inference. |
| --image-path | Path of the input image for inference.                       |
| --device     | Alternative infer device, cpu or cuda, default is cpu.       |

📛**Note**

If you want to **visualize** the attention mechanism, run the following command and the results will be saved in the **figures/heatmap** folder.

```bash
python utils/cam.py --image-path imgs/pneumonia_img1.png
```

More information about CAM can be found [here](https://github.com/zhoubolei/CAM)！💖

### Generation part

🚀Quick start, and the results will be saved in the **figures/generator_torch** folder.

```bash
python infer_generator.py --ckpt-path pretrained/sn-dcgan.pth --batch-size 1 --mode -1
```

📛**Note**

If you want to generate fake images for training or sprite images, run following commands:

* Generate a Sprite map. 【save results in **outs/sprite**】

  ```bash
  python infer_generator.py --ckpt-path pretrained/sn-dcgan.pth --batch-size 64 --mode 0
  ```

* Generate a batch of images. 【save results in **outs/singles**】

  ```bash
  python infer_generator.py --ckpt-path pretrained/sn-dcgan.pth --batch-size 50 --mode 1
  ```

  💨When you generate a batch of images, **batch-size** is whatever you like❤



## Evaluate

### Classification part

```bash
python eval.py --ckpt-path pretrained/resnet18-sam.pth
```

### Generation part

To evaluate a model, make sure you have **torch-fidelity** installed in requirements.txt❗

Then, you should prepare **two datasets**❗

* training datasets in **data/merge** folder. 【real images】
* generation datasets in **outs** folder. 【fake images】

----

Everything is ready, you can execute the following command：

```bash
fidelity --gpu 0 --fid --input1 data/merge --input2 data/outs/singles
```

More information about fidelity can be found [here](https://github.com/mseitzer/pytorch-fid)！💖



## Train

### Classification part

```bash
python train_classifier.py
```

💝 **More details about training your own dataset**

Please refer to **data/config.yaml** and [README.md](https://github.com/MaitreChen/MedGAN-ResLite/blob/main/configs/README.md).

In addition, you need to set the normalized parameters **mean** and **std**! Please refer to **utils/image_utils.py**. 

### Generation part

```bash
python train_dcgan.py
```



## Export

If you want to export the ONNX model for  **ONNXRuntime** or **OpenVINO**, please refer to [README.md](https://github.com/MaitreChen/MedGAN-ResLite/blob/main/onnx/export_onnx.py)!



## Deploy

To use ONNXRuntime, refer to [README.md](https://github.com/MaitreChen/MedGAN-ResLite/blob/main/onnx/README.md) and **onnx/inference_onnx.py**!

To use OpenVINO, refer to [README.md](https://github.com/MaitreChen/MedGAN-ResLite/blob/main/openvino/README.md)!



# 🌞Results

....



# 📞Contact

For any questions or suggestions about this project, welcome everyone to raise **issues**!

Also, please feel free to contact [hbchenstu@outlook.com](mailto:hbchenstu@outlook.com).

Thank you, wish you have a pleasant experience~~💓🧡💛💚💙💜
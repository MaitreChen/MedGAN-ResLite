<div align="center">
  <a href="" target="_blank">
  <img width="50%" src="https://github.com/MaitreChen/MedGAN-ResLite/blob/main/figures/logo.png"></a>
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

* Coming Back！

* ✅ Mar 10, 2024. The dataset was preprocessed using [Chest X-ray 2017](https://data.mendeley.com/datasets/rscbjbr9sj/2) with reference to MedMNIST [[paper](https://arxiv.org/abs/2110.14795)] practices.
* ✅ Mar 11, 2024. Train GAN & CNN again！
* ✅ Mar 13, 2024. Histogram equalization was tried, but it did not work well~
* ✅ Mar 15, 2024. Attempts were made to introduce residual connection in GAN, but the effect was not good and the training  speed was affected~
* ✅ Mar 20, 2024. Trying the WGAN training strategy and introducing Wasserstein distance did not work well~
* ✅ Mar 24, 2024. Add Pruning Sample by NNI.
* ✅ May 15, 2024. Release **PulmoInsight Web Applicaiton**！
* ✅ May 21, 2024. Release of [**MedGAN-ResLite-V2**](https://github.com/MaitreChen/MedGAN-ResLite/tree/v2)！

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

### Dataset

You can download dataset from this [link](https://drive.google.com/drive/folders/1DOG37hN4LZ8rwj1Mfxlt9qcZwaQetX2S?usp=share_link). It includes the pneumoniamnist original real dataset and the fake dataset synthesized using GAN (see data [README.md](https://github.com/MaitreChen/MedGAN-ResLite/blob/main/data/README.md) for details)

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

### Pretrained Checkpoints

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

### Performance comparison of different GAN

<div style="text-align: center;">
  <table>
    <tr>
      <th>Method</th>
      <th>Inception Score</th>
      <th>FID</th>
      <th>KID</th>
    </tr>
    <tr>
      <td style="text-align: center;">GAN</td>
      <td style="text-align: center;">2.20</td>
      <td style="text-align: center;">260.15</td>
      <td style="text-align: center;">0.42</td>
    </tr>
    <tr>
      <td style="text-align: center;">DCGAN</td>
      <td style="text-align: center;">2.20</td>
      <td style="text-align: center;">259.72</td>
      <td style="text-align: center;">0.39</td>
    </tr>
    <tr>
      <td style="text-align: center;">SH-DCGAN</td>
      <td style="text-align: center;">2.20</td>
      <td style="text-align: center;">206.14</td>
      <td style="text-align: center;">0.31</td>
    </tr>
  </table>
</div>


<div style="display:flex;flex-wrap:wrap;justify-content:center;">
    <div style="text-align:center;margin:10px;">
        <a href="" target="_blank">
            <img src="https://github.com/MaitreChen/MedGAN-ResLite/blob/main/figures/display/original.png" width="50%">
        </a>
        <br>
        <em>Original</em>
    </div>
    <div style="text-align:center;margin:10px;">
        <a href="" target="_blank">
            <img src="https://github.com/MaitreChen/MedGAN-ResLite/blob/main/figures/display/sh-dcgan.png" width="50%">
        </a>
        <br>
        <em>SH-DCGAN</em>
    </div>
</div>



### *Ablation study*

<table style="margin: 0 auto;">
  <thead>
    <tr>
      <th style="text-align:center;">Method</th>
      <th style="text-align:center;">Inception Score</th>
      <th style="text-align:center;">FID</th>
      <th style="text-align:center;">KID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:center;">DCGAN</td>
      <td style="text-align:center;">2.20</td>
      <td style="text-align:center;">259.72</td>
      <td style="text-align:center;">0.39</td>
    </tr>
    <tr>
      <td style="text-align:center;">DCGAN + Hinge</td>
      <td style="text-align:center;">2.20</td>
      <td style="text-align:center;">252.42</td>
      <td style="text-align:center;">0.38</td>
    </tr>
    <tr>
      <td style="text-align:center;">DCGAN + SN</td>
      <td style="text-align:center;">2.20</td>
      <td style="text-align:center;">232.59</td>
      <td style="text-align:center;">0.35</td>
    </tr>
    <tr>
      <td style="text-align:center;">SH-DCGAN</td>
      <td style="text-align:center;">2.20</td>
      <td style="text-align:center;">206.14</td>
      <td style="text-align:center;">0.31</td>
    </tr>
  </tbody>
</table>



### Performance comparison before and after improvement

<div align="center">
  <a href="" target="_blank">
  <img width="100%" src="https://github.com/MaitreChen/MedGAN-ResLite/blob/main/figures/display/res.png"></a>
</div>




### *Comparison of different CNN models*

<table style="margin: 0 auto;">
  <thead>
    <tr>
      <th style="text-align:center;">Model</th>
      <th style="text-align:center;">Accuracy/%</th>
      <th style="text-align:center;">Precision/%</th>
      <th style="text-align:center;">Recall/%</th>
      <th style="text-align:center;">F1 score/%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:center;">AlexNet</td>
      <td style="text-align:center;">90.16</td>
      <td style="text-align:center;">90.16</td>
      <td style="text-align:center;">90.16</td>
      <td style="text-align:center;">90.16</td>
    </tr>
    <tr>
      <td style="text-align:center;">VGG16</td>
      <td style="text-align:center;">91.22</td>
      <td style="text-align:center;">92.23</td>
      <td style="text-align:center;">91.22</td>
      <td style="text-align:center;">91.17</td>
    </tr>
    <tr>
      <td style="text-align:center;">VGG19</td>
      <td style="text-align:center;">91.76</td>
      <td style="text-align:center;">92.70</td>
      <td style="text-align:center;">91.76</td>
      <td style="text-align:center;">91.71</td>
    </tr>
    <tr>
      <td style="text-align:center;">ResNet34</td>
      <td style="text-align:center;">92.55</td>
      <td style="text-align:center;">93.26</td>
      <td style="text-align:center;">92.55</td>
      <td style="text-align:center;">92.52</td>
    </tr>
    <tr>
      <td style="text-align:center;">ResNet50</td>
      <td style="text-align:center;">91.15</td>
      <td style="text-align:center;">92.44</td>
      <td style="text-align:center;">92.15</td>
      <td style="text-align:center;">92.14</td>
    </tr>
    <tr>
      <td style="text-align:center;">MobileNetV2</td>
      <td style="text-align:center;">92.29</td>
      <td style="text-align:center;">92.60</td>
      <td style="text-align:center;">92.29</td>
      <td style="text-align:center;">92.27</td>
    </tr>
    <tr>
      <td style="text-align:center;">ResNet18</td>
      <td style="text-align:center;">92.02</td>
      <td style="text-align:center;">92.02</td>
      <td style="text-align:center;">92.02</td>
      <td style="text-align:center;">92.02</td>
    </tr>
    <tr>
      <td style="text-align:center;">ResNet18-SAM</td>
      <td style="text-align:center;"><strong>93.48</strong></td>
      <td style="text-align:center;"><strong>93.82</strong></td>
      <td style="text-align:center;"><strong>93.48</strong></td>
      <td style="text-align:center;"><strong>93.47</strong></td>
    </tr>
  </tbody>
</table>



### Interpretability

<div align="center">
  <a href="" target="_blank">
  <img width="100%" src="https://github.com/MaitreChen/MedGAN-ResLite/blob/main/figures/display/CAM.png"></a>
</div>






# 📞Contact

For any questions or suggestions about this project, welcome everyone to raise **issues**!

Also, please feel free to contact [hbchenstu@outlook.com](mailto:hbchenstu@outlook.com).

Thank you, wish you have a pleasant experience~~💓🧡💛💚💙💜
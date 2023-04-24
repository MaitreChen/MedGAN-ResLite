# 📣Introduction

This is a pneumonia classification project that utilizes generative adversarial networks to generate images of minority class samples, thereby addressing the issue of imbalanced class distribution and enhancing the classifier's generalization performance.

----

🚩 **New Updates**

* ✅ Mar 21, 2023. Creat "MedGAN-ResLite" project repository and Find MedMNIST.
* ✅ Mar 22, 2023. Generate pneumonia samples with DCGAN.
* ✅❗ May 23, 2023. Try Muti-Scale Fusion.
* ✅❗ May 25, 2023. Introduce class information into DCGAN to generate samples.【cDCGAN】
* ❓Replace original Loss function with 0Wasserstein distance.
* ✅ May 30, 2023. Replace original Loss function with Hinge Adversial Loss.
* ✅ Apri 1, 2023.  DCGAN + Spectral Normalization.
* ✅ Apri 4, 2023.  Add DCGAN metrics：Inception Score + FID + KID; Fuse and Split dataset;
* ✅ Apri 5, 2023.  Override the dataset inheritance class.
* ✅ Apri 6, 2023.  Write train, eval and infer scripts for classifier. And get a new-model by modifing input & output shape of pre-trained model. Add metrics：acc + auc + f1 + confusion matrix.
* ✅ April 7, 2023. Add scripts: export_onnx.py  and inference_onnx.py.
* ✅ April 8, 2023. Tuning the hyperparameters of DCGAN.
* ✅ April 14, 2023. Abalation Study: GAN, DCGAN, DCGAN+Hinge, DCGAN + SN, DCGAN + SH.

* ❓ Pruning：one-hot + iterative ，including L1✅、L2✅、FPGM✅、BNScale.
* ❓ Build the pruned model automatically.
* ❓ Knowledge distillation：design lightweight network A，and use pruned-model to guide A.
* ❓ Deploy model on CPU and NSC2 using OpenVINO，add Python ✅and C++ version.
* ❓ Consider：deploy on the web side using Django or flask.





# ✨Quick Start

.....










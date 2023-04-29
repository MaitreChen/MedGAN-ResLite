# ResNet18-SAM + OpenVNO🎇

## 📣Introdution

This is a folder for **openvino** model convert and inference!

## ❗Notice

* Make sure you have OpenVINO~=2022.3.0 installed!
* Please prepare the model file in ONNX format !
* Make sure the ONNX model is in the **"./pretrained/"** file and the converted OpenVINO models **(.xml + .bin)** are in **"./openvino/ir_models/"**!

## ✨Usage

### Options

#### 1 Convert Part

You can refer to the example in openvino document, here is the [link](https://docs.openvino.ai/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

Also, you can run the script to see how options are used by:

```bash
python mo.py -h
```

#### 2 Inference Part

| Option       | Description                                                  |
| ------------ | ------------------------------------------------------------ |
| --model-path | Model file path. This parameter is required and must be of type string, with a default value of 'openvino/ir_models/resnet18-sam.xml'. |
| --image-path | Image file path. This parameter is required and must be of type string, with a default value of 'imgs/normal_img1.png'. |
| --visualize  | Enable visualization. This parameter is optional, with a default value of True. When set to True, the visualization feature will be enabled. |
| --mode       | Inference mode. This parameter is optional, with a default value of 0. A value of 0 represents synchronous mode, while a value of 1 represents asynchronous mode. |
| --device     | Inference hardware. This parameter is optional, with a default value of CPU. The available options are CPU, GPU, and MYRAID. |



### Example

1. Download the **ResNet18-SAM** ONNX model from this [link](https://drive.google.com/file/d/1Z1d4VL_K6Tzq-INcHn85JVUPjKiYzlCI/view?usp=share_link) or refer to the **"./onnx/"** folder for export your onnx file.

2. Convert pth model to IR model:

   ```bash
   mo --input_model .\pretrained\resnet18-sam.onnx --output_dir .\openvino\ir_models\
   ```

3. Inference the IR model:

   ```bash
   python .\openvino\inference_openvino.py --model-path .\openvino\ir_models\resnet18-sam.xml --image-path .\imgs\normal_img1.png
   ```

   **CPU** is enabled by default to perform **synchronous** inference!



















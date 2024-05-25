# ResNet18-SAM + ONNX🎆

📣This is a folder for onnx model export and inference!

🧨Note, make sure you're working in the `root directory`!

----

✨The usage is as follows:

1. Download the pre-trained **ResNet18-SAM** from this [link](https://drive.google.com/file/d/1OAUln7TRDNdDi0nGrjqSCJgflFr5SWsF/view?usp=share_link) and put it into the folder `pretrained/`.【Of course, you can also train a model yourself!】

2. Export the ONNX Model:

   ```bash
   python onnx/export_onnx.py --ckpt-path pretrained/resnet18-sam.pth --output-path onnx/resnet18-sam.onnx
   ```

3. Inference the ONNX model:

   ```bash
   python onnx/inference_onnx.py --model-path onnx/resnet18-lite.onnx --image-path imgs/normal1.jpeg
   ```




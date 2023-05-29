from matplotlib import pyplot as plt
from openvino.runtime import Core
from time import time
import numpy as np
import cv2 as cv
import argparse
import os


def inference_resnet18sam(model_path, image_path, visualize, device):
    # Step 1. Load and Compile the Model
    ie = Core()
    model = ie.read_model(model=model_path)
    compiled_model = ie.compile_model(model=model, device_name=device)
    output_layer = compiled_model.output(0)

    # Step 2. Load and preprocess image
    image = cv.cvtColor(cv.imread(filename=image_path), code=cv.COLOR_BGR2GRAY)

    # resize to ResNet-18-SAM image shape
    input_image = cv.resize(src=image, dsize=(224, 224))

    # reshape to model input shape
    input_image = np.expand_dims([input_image], 0)

    # Step 3. Do inference and get result
    classes = ('normal', 'pneumonia')

    start_time = time()
    result_infer = compiled_model([input_image])[output_layer]
    end_time = time()
    print(f"Inference time: {end_time - start_time:.6f} ms")

    result_index = np.argmax(result_infer)
    out = classes[result_index]

    # visualization of result
    if visualize:
        plt.text(0.05, 0.95, str(out), fontsize=18,
                 verticalalignment='top', color='red')
        plt.imshow(image, 'gray')
        plt.show()


if __name__ == "__main__":
    # Define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, default='openvino/ir_models/resnet18-sam.xml',
                        help='path of the IR model (.xml) that will be inferred')
    parser.add_argument('--image-path', type=str, required=True, default='imgs/normal_img1.png',
                        help='path of the image that will be input')
    parser.add_argument('--visualize', type=bool, required=False, default=True,
                        help='Enabling visualization. (default: True)')
    parser.add_argument('--device', type=str, required=False, default='CPU',
                        help="Inference hardware. [CPU,GPU,MYRAID]  (default: CPU)")
    args = parser.parse_args()

    # Check input arguments
    if not os.path.exists(args.model_path):
        print(f'Cannot find the onnx model: {args.model_path}')
        exit()
    if not os.path.exists(args.image_path):
        print(f'Cannot find the input image: {args.image_path}')
        exit()

    # Do inference
    inference_resnet18sam(args.model_path, args.image_path,
                          args.visualize, args.device)

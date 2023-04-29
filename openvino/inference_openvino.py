from time import time
import numpy as np
import cv2 as cv
import argparse
import os

import openvino.runtime as ov

classes = ('normal', 'pneumonia')


def inference_resnet18sam(model_path, img_path, visualize, mode, device):
    print(f"Device on: {device}")
    # Step 1. Create OpenVINORuntime Core
    core = ov.Core()

    # Step 2. Compile the Model
    compiled_model = core.compile_model(model_path, device)

    # Step 3. Create an Inference Request
    infer_request = compiled_model.create_infer_request()

    # Get input shape and element type from the model
    input_tensor = infer_request.get_input_tensor()
    tensor_shape = input_tensor.get_shape()
    h, w = tensor_shape[2], tensor_shape[3]

    # Step 4. Set Inputs
    src = cv.imread(img_path, 0)
    resized_img = cv.resize(src, (w, h))
    img = np.array([np.array([resized_img])], dtype=np.float32)  # uint8 --> float32

    input_tensor = ov.Tensor(array=img, shared_memory=True)
    infer_request.set_input_tensor(input_tensor)

    # Step 5. Start Synchronous Inference
    mode_map_info = {0: 'synchronous', 1: 'asynchronous'}
    if mode_map_info[mode] == 'synchronous':
        start_time = time()
        infer_request.infer()
        end_time = time()
    elif mode_map_info[mode] == 'asynchronous':
        start_time = time()
        infer_request.start_async()
        infer_request.wait()
        end_time = time()
    print(f"Inference time: {end_time - start_time:.6f} ms")

    # Step 5. Get output and post-process
    output = infer_request.get_output_tensor()
    output_buffer = output.data
    res = classes[np.argmax(output_buffer).item()]
    print(f"The result : {res}")

    # visualization of result
    if visualize:
        resized_src = cv.resize(src, None, fx=10, fy=10)
        cv.putText(resized_src, "Prediction:" + str(res), (0, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv.imshow("Result", resized_src)
        cv.waitKey()
        cv.destroyAllWindows()


if __name__ == "__main__":
    # Define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, default='openvino/ir_models/resnet18-sam.xml')
    parser.add_argument('--image-path', type=str, required=True, default='imgs/normal_img1.png')
    parser.add_argument('--visualize', type=bool, required=False, default=True,
                        help='Enabling visualization. (default: True)')
    parser.add_argument('--mode', type=int, required=False, default=0,
                        help='Inference mode. [0 | 1] represents [synchronous| asynchronous]  (default: 0)')
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

    # Run to inference
    inference_resnet18sam(args.model_path, args.image_path, args.visualize, args.mode, args.device)

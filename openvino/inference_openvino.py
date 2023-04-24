import argparse

import openvino.runtime as ov
from time import time
import numpy as np
import cv2 as cv

classes = ('normal', 'pneumonia')


def inference_mnist(model_path, img_path, mode, device):
    print(f"Inference device: {device}")
    # Step 1. Create OpenVINORuntime Core
    core = ov.Core()

    # Step 2. Compile the Model
    compiled_model = core.compile_model(model_path, device)

    # Step 3. Create an Inference Request
    infer_request = compiled_model.create_infer_request()

    # Get input shape and element type from the model
    input_tensor = infer_request.get_input_tensor()
    # tensor_type = input_tensor.get_element_type()
    tensor_shape = input_tensor.get_shape()
    h, w = tensor_shape[2], tensor_shape[3]

    # Step 4. Set Inputs
    src = cv.imread(img_path, 0)
    resized_img = cv.resize(src, (w, h))
    img = np.array([np.array([resized_img])], dtype=np.float32)  # uint8 --> float32

    input_tensor = ov.Tensor(array=img, shared_memory=True)
    infer_request.set_input_tensor(input_tensor)

    # 5. Start Synchronous Inference
    if mode == 'sync':
        start_time = time()
        infer_request.infer()
        end_time = time()
    else:
        start_time = time()
        infer_request.start_async()
        infer_request.wait()
        end_time = time()
    print(f"Inference time: {end_time - start_time:.6f} ms")

    # 6. Get output and process
    output = infer_request.get_output_tensor()
    output_buffer = output.data
    res = classes[np.argmax(output_buffer).item()]
    print(f"The result : {res}")

    cv.putText(src, "Prediction:" + str(res), (0, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv.imshow("Result", src)
    cv.waitKey()
    cv.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--img', type=str, default='')
    parser.add_argument('--mode', default='sync', type=str, required=False)
    parser.add_argument('--device', type=str, required=True, help="[CPU,GPU,MYRAID]")
    args = parser.parse_args()

    model_path = args.model
    img_path = args.img
    mode = args.mode
    device = args.device

    inference_mnist(model_path, img_path, mode, device)

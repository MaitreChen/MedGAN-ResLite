from time import time
import cv2 as cv
import numpy as np
import argparse
import os

import onnxruntime

classes = ('normal', 'pneumonia')


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def get_test_transform():
    from torchvision import transforms
    return transforms.ToTensor()


def inference_resnet18sam(model_path, img_path, image_size=224):
    # initialize session
    session = onnxruntime.InferenceSession(model_path)
    session.set_providers(['CPUExecutionProvider'])

    # read image
    src = cv.imread(img_path, 0)

    # pre-process
    resized_img = cv.resize(src, (image_size, image_size))
    img = get_test_transform()(resized_img)
    img = img.unsqueeze_(0)

    # prepare inputs and get prediction
    inputs = {session.get_inputs()[0].name: to_numpy(img)}
    start_time = time()
    outs = session.run(None, inputs)
    end_time = time()

    # post-process
    res = classes[np.argmax(outs).item()]
    print(f"Inference time: {1000. * (end_time - start_time):.4f} ms")
    print(f"The result: {res}")

    # visualization of result
    resized_src = cv.resize(src, None, fx=10, fy=10)
    cv.putText(resized_src, "Prediction:" + str(res), (0, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv.imshow("result", resized_src)
    cv.waitKey()
    cv.destroyAllWindows()


if __name__ == "__main__":
    # Define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, default='pretrained/resnet18-sam.onnx')
    parser.add_argument('--image-path', type=str, required=True, default='imgs/normal_img1.png')
    args = parser.parse_args()

    # Check input arguments
    if not os.path.exists(args.model_path):
        print(f'Cannot find the onnx model: {args.model_path}')
        exit()
    if not os.path.exists(args.image_path):
        print(f'Cannot find the input image: {args.image_path}')
        exit()

    # Run to inference
    inference_resnet18sam(args.model_path, args.image_path)

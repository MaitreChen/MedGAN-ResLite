import onnxruntime
import cv2 as cv
import numpy as np

from time import time
import argparse

classes = ('normal', 'pneumonia')


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def get_test_transform():
    from torchvision import transforms
    return transforms.ToTensor()


def inference_mnist(model_path, img_path, image_size=224):
    session = onnxruntime.InferenceSession(model_path)
    session.set_providers(['CPUExecutionProvider'])
    src = cv.imread(img_path, 0)
    resized_img = cv.resize(src, (image_size, image_size))
    img = get_test_transform()(resized_img)
    img = img.unsqueeze_(0)

    inputs = {session.get_inputs()[0].name: to_numpy(img)}
    start_time = time()
    outs = session.run(None, inputs)
    end_time = time()

    # post-process
    res = classes[np.argmax(outs).item()]
    print(f"Inference time: {1000. * (end_time - start_time):.4f} ms")
    print(f"The result: {res}")

    # cv.putText(src, "Prediction:" + str(res), (0, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    # cv.imshow("result", src)
    # cv.waitKey()
    # cv.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str, default='./best.onnx')
    # parser.add_argument('--model', type=str, default='./original_resnet.onnx')
    parser.add_argument('--img', type=str, default='../imgs/test4_normal.png')
    args = parser.parse_args()

    # model_path = args.model
    img_path = args.img
    model_path = './best.onnx'
    inference_mnist(model_path, img_path)

    model_path = './original_resnet.onnx'
    inference_mnist(model_path, img_path)

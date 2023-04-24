import argparse

import torch

from models.resnet import ResNet, BasicBlock


def export_onnx(model, ckpt_path, output_dir):
    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()

    # export onnx
    dummy_input = torch.randn(1, 1, 224, 224).cuda()
    torch.onnx.export(model, dummy_input, output_dir, input_names=['input'],
                      output_names=['output'], verbose=True,
                      opset_version=11)

    print("Successfully exporting onnx model!")


if __name__ == "__main__":
    # Define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--ckpt-path', type=str,
                        default='../checkpoints/cls/exp_fake_scratch_pretrained_test/pneumonia_best.pth')
    parser.add_argument('--output-path', type=str, default='./original_resnet.onnx')

    args = parser.parse_args()

    # Check input argument
    ckpt_path = args.ckpt_path
    output_path = args.output_path

    # Build model
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=2).cuda()

    export_onnx(model, ckpt_path, output_path)

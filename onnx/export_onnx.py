import argparse
import os

import torch

from models.resnet import ResNet, BasicBlock


def export_onnx(ckpt_path, output_dir):
    # Build model and load checkpoints
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=2).cuda()
    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()

    # Prepare input
    dummy_input = torch.randn(1, 1, 224, 224).cuda()

    # Export onnx
    torch.onnx.export(model, dummy_input, output_dir, input_names=['input'],
                      output_names=['output'], verbose=True,
                      opset_version=11)

    print("Successfully exporting onnx model!")


if __name__ == "__main__":
    # Define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', type=str, required=True, default='./pretrained/cnn/resnet18-sam.pth',
                        help='path of the checkpoint that will be exported')
    parser.add_argument('--output-path', type=str, required=True, default='./onnx/resnet18-sam.onnx',
                        help='path for saving the ONNX model')
    args = parser.parse_args()

    # Check input arguments
    if not os.path.exists(args.ckpt_path):
        print(f'Cannot find checkpoint path: {args.ckpt_path}')
        exit()

    # Run to export
    export_onnx(args.ckpt_path, args.output_path)

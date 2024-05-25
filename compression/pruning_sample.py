"""
# This script demonstrates how to perform model pruning using NNI.
# In this script, we load a ResNet-18 model, and apply FPGM pruning to it, and then export the pruned model
# to the ONNX format for deployment or further analysis.
"""
import argparse
import os
import sys
import torch

from nni.compression.pytorch.utils import count_flops_params
from nni.compression.pytorch.pruning import LevelPruner, L1NormPruner, L2NormPruner, FPGMPruner
from nni.compression.pytorch import ModelSpeedup

from models.create_model import create_cnn_model

sys.path.append('../')


def prune_and_export_model(model_name, checkpoints_path, output_onnx_path, pruning_ratio=0.6, pruning_method='l1'):
    """
    Prune the specified model using NNI and export the pruned model to ONNX format.

    Args:
        model_name (str): Name of the model architecture to be used (e.g., 'resnet18').
        checkpoints_path (str): Path to the pretrained model's state dictionary.
        output_onnx_path (str): Path to save the pruned model in ONNX format.
        pruning_ratio (float, optional): Ratio of weights to prune. Defaults to 0.6.
        pruning_method (str) : Pruning method to use. (e.g., 'l1','l2','level','fpgm')
    """
    # Define the configuration list for pruning
    config_list = [{'sparsity': pruning_ratio, 'op_types': ['Conv2d']}]

    # Create the original model
    model = create_cnn_model(model_name, pretrained=False)

    # Load the pretrained model's state dictionary
    state_dict = torch.load(checkpoints_path)
    model.load_state_dict(state_dict)

    # Create a dummy input tensor for model analysis
    dummy_input = torch.rand(1, 1, 224, 224).to('cpu')

    # Calculate FLOPs and parameters of the original model
    flops1, params1, _ = count_flops_params(model, dummy_input, verbose=True)
    print(f"\nOriginal Model:\nFLOPs {flops1 / 1e6:.2f}M, Params {params1 / 1e6:.2f}M")

    # Initialize the Pruner with the model and configuration
    pruner = None
    if pruning_method == 'l1':
        pruner = L1NormPruner(model, config_list)
    elif pruning_method == 'l2':
        pruner = L2NormPruner(model, config_list)
    elif pruning_method == 'level':
        pruner = LevelPruner(model, config_list)
    elif pruning_method == 'fpgm':
        pruner = FPGMPruner(model, config_list)

    # Perform pruning and obtain the pruned model and masks
    _, masks = pruner.compress()

    # Remove the pruning-related hooks and modules from the model
    pruner._unwrap_model()

    # Speed up the pruned model
    ModelSpeedup(model, dummy_input, masks).speedup_model()

    # Calculate FLOPs and parameters of the original model
    flops2, params2, _ = count_flops_params(model, dummy_input, verbose=True)
    print(f"\nPruned Model:\nFLOPs {flops2 / 1e6:.2f}M, Params {params2 / 1e6:.2f}M")

    # Export the pruned model to ONNX format
    torch.onnx.export(model, dummy_input, output_onnx_path, input_names=['input'],
                      output_names=['output'], verbose=True, opset_version=11)

    print("Successfully exported ONNX model!")


if __name__ == '__main__':
    # Define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, required=True, default='resnet18',
                        help='')
    parser.add_argument('--ckpt-path', type=str, required=True, default='../pretrained/resnet18.pth',
                        help='path of the checkpoint that will be pruned and exported')
    parser.add_argument('--output-path', type=str, required=True, default='resnet18-lite-test.onnx',
                        help='path for saving the ONNX model')
    parser.add_argument('--pruning-ratio', type=int, required=True, default=0.6,
                        help='')
    parser.add_argument('--pruning-mathod', type=str, required=True, default='l1',
                        help='Pruning Method [ l1 | l2 | level | fpgm ]')

    args = parser.parse_args()

    # Check input arguments
    if not os.path.exists(args.ckpt_path):
        print(f'Cannot find checkpoint path: {args.ckpt_path}')
        exit()

    # Run to prune and export
    prune_and_export_model(
        model_name=args.model_name,
        checkpoints_path=args.ckpt_path,
        output_onnx_path=args.output_path,
        pruning_ratio=args.pruning_ratio,
        pruning_method=args.pruning_method
    )

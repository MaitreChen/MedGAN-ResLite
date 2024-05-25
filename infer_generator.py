"""
Here is a script for sn-dcgan to generate image visualizations.
At the same time, you can use this script to generate minority class samples to train a classifier.
"""
import math

import cv2 as cv
import argparse
import os

import torchvision.utils as vutils
import torch

from utils.image_utils import decode_to_numpy
from utils.common_utils import create_folder

# from models.gan import Generator
# from models.dcgan import Generator
from models.sndcgan import Generator


def infer(ckpt_path, batch_size, z_dim, image_size, device, save_mode, save_dir='figures/generator_torch/'):
    """
    Perform inference using a pre-trained DCGAN model.

    Args:
        ckpt_path (str): Path to the checkpoints of DCGAN.
        batch_size (int): Number of samples to generate.
        z_dim (int): Dimension of the input noise vector (default 128).
        image_size (int): Size of the generated images (default 128).
        device (torch.device): Device to perform inference on.
        save_mode (int):
                  0: generate images for training;
                  1: save mini-batch images (Sprite map);
                 -1: generate a single image.
        save_dir (str, optional): Directory to save the inference results (default 'figures/generator_torch/').

    Returns:
        None
    """

    # Create random tensors
    fixed_z = torch.randn(batch_size, z_dim)
    fixed_z = fixed_z.view(fixed_z.size(0), fixed_z.size(1), 1, 1)

    # Define model and load checkpoint
    G = Generator(z_dim=z_dim, image_size=image_size)
    # G = Generator()
    G.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device)))
    G.to(device)
    G.eval()

    # Start Inference
    fake_images = G(fixed_z.to(device))

    # Generate & Save images
    if save_mode == -1:
        print("Generating a single image...")
        save_path = os.path.join(save_dir, 'single')
        create_folder(save_path)
        print(f"fake image save into {save_path}")

        fake_image_numpy, fake_image = decode_to_numpy(fake_images[0])
        fake_image = fake_image.transpose(1, 2, 0)
        cv.imwrite(os.path.join(save_path, 'fake.png'), fake_image)
    elif save_mode == 0:
        print("Generating a batch of images...")
        save_path = os.path.join(save_dir, 'images')
        create_folder(save_path)
        print(f"fake images save into {save_path}")

        idx = 0
        for i in range(batch_size):
            fake_image_numpy, fake_image = decode_to_numpy(fake_images[i][0])
            path = os.path.join(save_path, f'fake_img{idx}.png').replace('\\', '/')
            cv.imwrite(path, fake_image)
            print(path)
            idx += 1
    elif save_mode == 1:
        print("Generating a Sprite map...")
        save_path = os.path.join(save_dir, 'sprite')
        create_folder(save_path)
        print(f"fake images sprite save into {save_path}")

        # Normalize the images to the range [0, 1]
        fake_images = (fake_images - fake_images.min()) / (fake_images.max() - fake_images.min())
        grid_size = int(math.sqrt(batch_size))
        image_grid = vutils.make_grid(fake_images, nrow=grid_size, normalize=True)
        vutils.save_image(image_grid, os.path.join(save_path, 'fake_sprite.png'))
    print("Successfully!")


if __name__ == "__main__":
    # Define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', type=str, required=False, default='pretrained/gan/sh-dcgan.pth',
                        help='sn-dcgan checkpoints path for inference')
    parser.add_argument('--batch-size', type=int, required=False, default=1,
                        help='batch size for inference.')
    parser.add_argument('--image-size', type=int, default=128,
                        help='image size (default: 128)')
    parser.add_argument('--z-dim', type=int, default=100,
                        help='z-dim of generator (default: 128)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='inference device. [cuda | cpu]')
    parser.add_argument('--mode', type=int, default=-1, required=False,
                        help='generate images for [single:-1 | images:0 | sprite:1] (default: -1)')
    args = parser.parse_args()

    # Check input argument
    if not os.path.exists(args.ckpt_path):
        print(f'Cannot find the dcgan checkpoints: {args.ckpt_path}')
        exit()

    # Run to generate fake images
    infer(args.ckpt_path, args.batch_size, args.z_dim, args.image_size, args.device, args.mode)

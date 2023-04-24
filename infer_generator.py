"""
Here is a script for sn-dcgan to generate image visualizations.
At the same time, you can use this script to generate minority class samples to train a classifier.

Here are some tips if you want to generate minority class samples:
First, you can set mode like this distinguish between different situations
---------------------------------------------------------------
        set mode = 0 if batch_size == 1349 else 1
---------------------------------------------------------------
Where:
        1349 means generate single images for classifier and evaluation;
        64 means generate a sprite image for visualization;

Secondly, you can create folder to save single images or sprite
---------------------------------------------------------------
            save_path = f'outs/exp_{lab_name}/'
            create_folder(save_path + 'single/')
            create_folder(save_path + 'sprite/')
---------------------------------------------------------------

Then, you can run this script!
---------------------------------------------------------------
    # Run to save
    infer(ckpt_path, batch_size, z_dim, image_size, save_path, save_mode=mode)
---------------------------------------------------------------

Finally, you can get what you want in direct;
"""

from matplotlib import pyplot as plt
from time import time
import cv2 as cv
import argparse
import os

import torchvision
import torch

from utils.dcgan_dataset import get_dataloader_dcgan
from utils.image_utils import decode_to_numpy
from utils.common_utils import create_folder

# from models.gan import Generator
# from models.dcgan import Generator


from models.sndcgan import Generator


def infer(ckpt_path, batch_size, z_dim, image_size, save_path, save_mode):
    """
    :param ckpt_path: checkpoints of dcgan
    :param batch_size: #
    :param z_dim: default 128
    :param image_size: default 128
    :param save_path: #
    :param save_mode: 0:save single images; 1:save mini-batch images(Sprite chart)
    """
    out_channels = 1

    # Create random tensors
    fixed_z = torch.randn(batch_size, z_dim)
    fixed_z = fixed_z.view(fixed_z.size(0), fixed_z.size(1), 1, 1)
    print(f"fixed_z.shape={fixed_z.shape}")

    # Define model and load checkpoint
    G = Generator(z_dim=z_dim, image_size=image_size, out_channels=out_channels)
    G.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cuda')))
    G.to(device)
    G.eval()

    # Start Inference
    start_time = time()
    fake_images = G(fixed_z.to(device))
    end_time = time()
    print(f"Infer time: {end_time - start_time:.6f} ms")

    # save images
    if save_mode == 0:
        idx = 0
        for i in range(batch_size):
            fake_image_numpy, fake_image = decode_to_numpy(fake_images[i][0])
            path = os.path.join(save_path, 'single', f'fake_img{idx}.png')
            cv.imwrite(path, fake_image)
            idx += 1
    elif save_mode == 1:
        save_fake_path = f"{save_path}/sprite/fake.png"
        torchvision.utils.save_image(fake_images, save_fake_path)
    else:
        pass

    # Show fake images
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle("fake images", fontsize=34)
    for i in range(batch_size):
        plt.subplot(5, 5, i + 1)
        fake_image_numpy, fake_image = decode_to_numpy(fake_images[i][0])
        plt.imshow(fake_image_numpy, 'gray')
        plt.axis('off')

    plt.show()

    """
    # Create Dataset AND save real images
    train_dataloader = get_dataloader_dcgan(batch_size)
    batch_iterator = iter(train_dataloader)  # convert to iteration
    real_images = next(batch_iterator)  # get the first element
    torchvision.utils.save_image(real_images, 'real.png')
    
    # show real images
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle("real images", fontsize=34)
    for i in range(batch_size):
        plt.subplot(5, 5, i + 1)
        plt.imshow(decode_to_numpy(real_images[i][0])[0], 'gray')
        plt.axis('off')
    plt.show()
    """


if __name__ == "__main__":
    # Define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=str, default=25,
                        help='batch size for inference. (default: 25)')
    parser.add_argument('--image-size', type=int, default=64, help='image size (default: 128)')
    parser.add_argument('--z-dim', type=int, default=128, help='z-dim of generator (default: 128)')
    parser.add_argument('--ckpt-path', type=str, default='pretrained/sn-dcgan.pth',
                        help='sn-dcgan checkpoints path for inference')
    parser.add_argument('--use_gpu', action='store_true', default=True, help='turn on flag to use GPU')
    args = parser.parse_args()

    # Check input argument
    batch_size = args.batch_size
    image_size = args.image_size
    z_dim = args.z_dim
    ckpt_path = args.ckpt_path
    device = 'cuda' if args.use_gpu else 'cpu'

    # Run to see
    infer(ckpt_path, batch_size, z_dim, image_size, save_path=-1, save_mode=-1)

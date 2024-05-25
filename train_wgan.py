# IMPORT PACKAGES
from matplotlib import pyplot as plt
import argparse

import yaml
import os

import cv2 as cv

from pytorch_fid import fid_score
import torch.autograd as autograd
import torch.optim

from models.create_model import create_gan_model

from utils.dcgan_dataset import get_dataloader_dcgan
from utils.common_utils import create_folder, print_separator
from utils.image_utils import decode_to_numpy
from utils.setup import *


def train(G, D, dataloader, args, image_save_dir):
    print('==> Training started..')

    G.train()
    D.train()

    G.to(DEVICE)
    D.to(DEVICE)

    if args.model_name == 'wgan':
        g_optimizer = torch.optim.RMSprop(G.parameters(), lr=5e-5)
        d_optimizer = torch.optim.RMSprop(D.parameters(), lr=5e-5)
    elif args.model_name == 'wgan-gp':
        g_optimizer = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0, 0.9))
        d_optimizer = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0, 0.9))

    for epoch in range(1, args.num_epochs + 1):
        for batch_idx, real in enumerate(dataloader):
            real = real.to(DEVICE)

            mini_batch_size = real.size()[0]

            for p in D.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update

            d_loss_real = 0.0
            d_loss_fake = 0.0
            Wasserstein_D = 0.0

            # --------------------
            # 2. Update D network
            # --------------------

            for d_iter in range(CRITIC_ITERATIONS):
                D.zero_grad()

                # train with real
                real_data_v = torch.autograd.Variable(real)
                d_loss_real = D(real_data_v)
                d_loss_real = d_loss_real.mean()
                d_loss_real.backward(mone)

                # train with fake
                fixed_z = torch.randn(mini_batch_size, args.z_dim, 1, 1).to(DEVICE)
                with torch.no_grad():
                    fixed_z = torch.autograd.Variable(fixed_z)
                    fake = G(fixed_z).detach()

                d_loss_fake = D(fake)
                d_loss_fake = d_loss_fake.mean()
                d_loss_fake.backward(one)

                # train with gradient penalty
                gradient_penalty = calc_gradient_penalty(D, real_data_v.data, fake.data)
                gradient_penalty.backward()

                d_cost = d_loss_fake - d_loss_real + gradient_penalty
                Wasserstein_D = d_loss_real - d_loss_fake
                d_optimizer.step()
                print(f' loss_fake: {d_loss_fake}, loss_real: {d_loss_real}')

                if args.model_name == 'wgan':
                    for p in D.parameters():
                        p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

            for p in D.parameters():
                p.requires_grad = False  # to avoid computation

            G.zero_grad()

            # --------------------
            # 2. Update G network
            # --------------------
            fixed_z = torch.randn(args.batch_size, args.z_dim, 1, 1).to(DEVICE)
            fixed_z = torch.autograd.Variable(fixed_z)
            fake = G(fixed_z)
            g_loss = D(fake).mean()
            g_loss.backward(mone)
            g_cost = -g_loss
            g_optimizer.step()

            print_separator()
            print(f' g_cost: {g_cost} || d_cost: {d_cost} || Wasserstein_D: {Wasserstein_D}')

        print(f'Epoch: {epoch} / {args.num_epochs}')
        if epoch % args.save_interval == 0:
            # # Save model
            ckpt_path = os.path.join(ckpt_save_dir, f'epoch{epoch}.pth')
            torch.save(G.state_dict(), ckpt_path)

            # Test generator
            fig = plt.figure(figsize=(5, 5))
            fig.suptitle(f"epoch {epoch}", fontsize=17)
            with torch.no_grad():
                input_z = torch.randn(args.batch_size // 2, args.z_dim).to(DEVICE)
                input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
                fake = G(input_z).detach().cpu()
                for i in range(0, 25):
                    plt.subplot(5, 5, i + 1)
                    plt.imshow(fake[i][0].cpu().detach().numpy(), 'gray')
                    plt.axis('off')
                plt.show()

            num_fake_images = 2534
            with torch.no_grad():
                # Create random tensors for computing fid value
                fixed_z = torch.randn(num_fake_images, args.z_dim)
                fixed_z = fixed_z.view(fixed_z.size(0), fixed_z.size(1), 1, 1)
                # G.eval()
                fake_images_ = G(fixed_z.to(DEVICE))
                print("Generating a batch of images...")

                image_save_path = os.path.join(image_save_dir, 'epoch' + str(epoch))
                create_folder(image_save_path)

                idx = 0
                for i in range(num_fake_images):
                    fake_image_numpy, fake_image_ = decode_to_numpy(fake_images_[i][0])
                    path = os.path.join(image_save_path, f'fake_img{idx}.png')
                    cv.imwrite(path, fake_image_)
                    idx += 1
                print("Starting computing fid value...")
                real_images_folder = 'data/chest_xray2017_size64_train_val_merge'
                generated_images_folder = image_save_path

                fid_value = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder],
                                                                batch_size=256, device='cuda', dims=2048)
                print("\033[96mFID value: {:.2f}\033[0m".format(fid_value))
                log_fid_value(epoch, fid_value)


def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(args.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(DEVICE)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(DEVICE)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(DEVICE),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def get_argparse():
    # Define cmd arguments
    parser = argparse.ArgumentParser()

    # DCGAN Options
    parser.add_argument('--image-size', type=int, default=128, help='image size (default: 128)')
    parser.add_argument('--z-dim', type=int, default=128, help='z-dim of generator (default: 100)')

    # Train Options
    parser.add_argument('--batch-size', type=int, default=64, help='batch size for training (default: 64)')
    parser.add_argument('--num-workers', type=int, default=24, help='number of data loading workers')
    parser.add_argument('--num_epochs', type=int, default=2000, help='number of epochs for training (default: 200)')
    parser.add_argument('--seed', type=int, default=3407, help='random seed 1234 | 47 | 3407')
    parser.add_argument('--use-gpu', action='store_true', default=True, help='turn on flag to use GPU')

    # Model Options
    parser.add_argument('--model_name', type=str, default='wgan', help='[wgan | wgan-gp]')

    # File Management Options
    parser.add_argument('--save-interval', type=int, default=5, help='interval of saving checkpoints')
    parser.add_argument('--exp-name', type=str, default='exp4_real_wgan ',
                        help='exp name for checkpoints directory')

    return parser


if __name__ == '__main__':
    args = get_argparse().parse_args()

    DEVICE = 'cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu'

    LAMBDA = 10
    CRITIC_ITERATIONS = 5
    WEIGHT_CLIP = 0.01

    one = torch.tensor(1, dtype=torch.float)
    mone = one * -1
    one = one.to(DEVICE)
    mone = mone.to(DEVICE)

    # Define directory paths
    CHECKPOINTS_DIR = 'checkpoints/wgan'
    LOG_DIR = f'logs/wgan/{args.exp_name}'
    CONFIG_PATH = './configs/config.yaml'
    IMAGE_SAVE_DIR = f'outs/wgan/{args.exp_name}'

    # Load YAML configuration
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        yaml_info = yaml.load(f.read(), Loader=yaml.FullLoader)
    n_channels = yaml_info['n_channels']
    DATA_DIR = yaml_info['root3']
    CLASS_NAME = 'normal'
    MEAN = yaml_info['mean']
    STD = yaml_info['std']

    # Set checkpoints dir
    ckpt_save_dir = os.path.join(CHECKPOINTS_DIR, args.exp_name)
    create_folder(ckpt_save_dir)

    # Set log info
    log_filename = f'{LOG_DIR}/{args.exp_name}/'
    create_folder(log_filename)
    setup_logging(log_filename)
    log_hyperparameters(args)

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Get dataloader
    train_dataloader = get_dataloader_dcgan(DATA_DIR, CLASS_NAME, args.batch_size, args.num_workers)

    # Initialize model and weight
    G, D = create_gan_model(args.model_name, z_dim=args.z_dim, image_size=args.image_size, in_out_channels=n_channels)

    # Train WGAN
    train(G, D, train_dataloader, args, IMAGE_SAVE_DIR)

# IMPORT PACKAGES
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from time import time
import numpy as np
import argparse
import random
import os
import yaml

# from torch.utils.tensorboard import SummaryWriter
import torchvision.utils
import torch.nn as nn
import torch

from utils.dcgan_dataset import get_dataloader_dcgan
from utils.common_utils import create_folder, print_info
from utils.plot_utils import plot_g_d_loss

from models.sndcgan import Generator, Discriminator


# from models.dcgan import Generator, Discriminator

def set_seed(seed):
    """
    Set random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Create a function to train the model
def train_model(G, D, dataloader, num_epochs, g_lr, d_lr, beta1, beta2):
    G.to(DEVICE)
    D.to(DEVICE)

    # Set train mode
    G.train()
    D.train()

    # Define loss function
    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    # Set up the optimization algorithm
    g_optimizer = torch.optim.Adam(G.parameters(), lr=g_lr, betas=(beta1, beta2))
    d_optimizer = torch.optim.Adam(D.parameters(), lr=d_lr, betas=(beta1, beta2))

    batch_size = dataloader.batch_size

    # Use hard-coded parameters
    mini_batch_size = batch_size

    # Set the iteration counter
    iteration = 1
    G_losses = []
    D_losses = []

    for epoch in range(1, num_epochs + 1):
        # Record start time
        t_epoch_start = time()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        for i, images in enumerate(dataloader):
            # --------------------
            # 1. Update D network: maximize logs(D(x)) + logs(1 - D(G(z)))
            # --------------------

            images = images.to(DEVICE)

            # Make real and fake labels: 1 & 0
            mini_batch_size = images.size()[0]
            label_real = torch.full((mini_batch_size,), 1.).to(DEVICE)
            label_fake = torch.full((mini_batch_size,), 0.).to(DEVICE)

            # Discriminate real images
            d_out_real = D(images)

            # Make and Discriminate fake images
            input_z = torch.randn(mini_batch_size, args.z_dim).to(DEVICE)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_images = G(input_z)
            d_out_fake = D(fake_images)

            # Compute loss of real and fake batches as D loss
            d_loss_real = None
            d_loss_fake = None
            if args.loss_mode == 'vanilla':
                d_loss_real = criterion(d_out_real.view(-1), label_real)
                d_loss_fake = criterion(d_out_fake.view(-1), label_fake)
            elif args.loss_mode == 'hinge':
                d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
                d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()

            d_loss = d_loss_real + d_loss_fake

            # Update D
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # --------------------
            # 2. Update G network: maximize logs(D(G(z)))
            # --------------------

            #  Make fake images
            input_z = torch.randn(mini_batch_size, args.z_dim).to(DEVICE)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_images = G(input_z)
            d_out_fake = D(fake_images)

            # Compute G loss
            g_loss = None
            if args.loss_mode == 'vanilla':
                g_loss = criterion(d_out_fake.view(-1), label_real)
            elif args.loss_mode == 'hinge':
                g_loss = -d_out_fake.mean()

            # Update G
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # --------------------
            # 3. Write down result
            # --------------------
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()

            # Save Losses for plotting later
            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())

            iteration += 1

        t_epoch_finish = time()

        # Plot the progress
        print_info()
        print(
            f'epoch {epoch} || Epoch_D_Loss:{epoch_d_loss / batch_size:.4f} || Epoch_G_Loss:{epoch_g_loss / batch_size:.4f}')
        print(f'Time taken: {t_epoch_finish - t_epoch_start:.3f} sec.')

        t_epoch_start = time()

        # If at save interval => save generated image samples and checkpoints
        if epoch % args.save_interval == 0:
            # # Save model
            ckpt_path = os.path.join(ckpt_save_dir, f'epoch{epoch}.pth')
            torch.save(G.state_dict(), ckpt_path)

            # Test generator
            fig = plt.figure(figsize=(5, 5))
            fig.suptitle(f"epoch {epoch}", fontsize=17)
            with torch.no_grad():
                input_z = torch.randn(batch_size // 2, args.z_dim).to(DEVICE)
                input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
                fake = G(input_z).detach().cpu()
                for i in range(0, 25):
                    plt.subplot(5, 5, i + 1)
                    plt.imshow(fake[i][0].cpu().detach().numpy(), 'gray')
                    plt.axis('off')
                plt.show()

        # # tensorboard visualization
        # writer.add_scalars(main_tag='Training Loss',
        #                    tag_scalar_dict={'d_loss': epoch_d_loss / batch_size,
        #                                     'g_loss': epoch_g_loss / batch_size},
        #                    global_step=epoch)

        torch.save(G.state_dict(), os.path.join(ckpt_save_dir, f'last.pth'))

    print('Training completed!')

    return G, D, G_losses, D_losses


def get_argparse():
    # Define cmd arguments
    parser = argparse.ArgumentParser()

    # DCGAN Options
    parser.add_argument('--image-size', type=int, default=64, help='image size (default: 128)')
    parser.add_argument('--z-dim', type=int, default=128, help='z-dim of generator (default: 100)')

    # Train Options
    parser.add_argument('--batch-size', type=int, default=64, help='batch size for training (default: 64)')
    parser.add_argument('--workers', type=int, default=12, help='number of data loading workers')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs for training (default: 200)')
    parser.add_argument('--g-lr', type=float, default=0.0001, help='learning rate of generator (default: 0.0001)')
    parser.add_argument('--d-lr', type=float, default=0.0004, help='learning rate of discriminator (default: 0.0004)')
    parser.add_argument('--beta1', type=float, default=0.0, help='beta of generator (default: 0.5)')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta of discriminator (default: 0.999)')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--use-gpu', action='store_true', default=True, help='turn on flag to use GPU')
    parser.add_argument('--loss-mode', type=str, default='vanilla', help='gan loss for training. [vanilla| hinge]')

    # File Management Options
    parser.add_argument('--save-interval', type=int, default=4, help='interval of saving checkpoints')
    parser.add_argument('--output-dir', type=str, default='fake_output', help='fake images output directory')
    parser.add_argument('--name', type=str, default='exp1',
                        help='exp name for checkpoints directory')

    return parser


if __name__ == "__main__":
    args = get_argparse().parse_args()

    with open('./configs/config.yaml', 'r', encoding='utf-8') as f:
        yaml_info = yaml.load(f.read(), Loader=yaml.FullLoader)
    n_channels = yaml_info['n_channels']

    DEVICE = 'cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu'

    # Set checkpoints dir
    print_info()
    print('==> Creating folders..')
    ckpt_save_dir = os.path.join("checkpoints/dcgan/", args.name)
    create_folder(ckpt_save_dir)

    # Set logs dir
    log_path = os.path.join('logs/dcgan/', args.name)
    create_folder(log_path)

    # Set fake images dir
    output_save_dir = os.path.join(args.output_dir, args.name)
    create_folder(output_save_dir)

    # Create the table object, name, and alignment
    table = PrettyTable(['Hyper-Parameters & data infos', 'Value'])
    table.align['Hyper-Parameters & data infos'] = 'l'
    table.align['Value'] = 'r'

    # Add to table
    table.add_row(['Batch size', args.batch_size])
    table.add_row(['Workers', args.workers])
    table.add_row(['Num epochs', args.epochs])
    table.add_row(['Generator LR', args.g_lr])
    table.add_row(['Discriminator LR', args.d_lr])
    table.add_row(['beta1', args.beta1])
    table.add_row(['beta2', args.beta2])
    table.add_row(['random seed', args.seed])
    table.add_row(['Device', DEVICE])
    table.add_row(["", ""])
    table.add_row(['z_dim', args.z_dim])
    table.add_row(['image size', args.image_size])
    table.add_row(['n_channels', n_channels])
    table.add_row(["", ""])
    table.add_row(['checkpoints path', ckpt_save_dir])
    table.add_row(['fake images path', output_save_dir])
    print(table)

    # # # Set event_save_path
    # tensorboard_dir = os.path.join('logs/dcgan', args.name)
    # create_folder(tensorboard_dir)
    #
    # # Initialize
    # writer = SummaryWriter(tensorboard_dir)
    # print(f"events save to: {tensorboard_dir}")

    set_seed(args.seed)

    # Get dataloader
    print_info()
    print('==> Getting dataloader..')
    train_dataloader = get_dataloader_dcgan(args.batch_size, args.workers)

    # Whether training form checkpoint
    print_info()
    print('==> Building model..')
    # Initialize model and weight
    G = Generator(z_dim=args.z_dim, image_size=args.image_size, out_channels=n_channels)
    D = Discriminator(image_size=args.image_size, in_channels=n_channels)

    # Train model
    print_info()
    print('==> Training model..')
    G_update, D_update, G_loss_set, D_loss_set = train_model(G, D, train_dataloader, args.epochs, args.g_lr, args.d_lr,
                                                             args.beta1,
                                                             args.beta2)

    # Training visualization
    plot_g_d_loss(G_loss_set, D_loss_set, ckpt_save_dir)

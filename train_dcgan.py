# IMPORT PACKAGES
from time import time
import argparse
import yaml
import os

from prettytable import PrettyTable
import cv2 as cv

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Adam
import torch.optim
from torch import nn

from torch.utils.tensorboard import SummaryWriter
from pytorch_fid import fid_score

from models.create_model import create_gan_model

from utils.dcgan_dataset import get_dataloader_dcgan
from utils.loss_functions import get_loss_function
from utils.common_utils import create_folder, print_separator
from utils.image_utils import decode_to_numpy
from utils.metrics import evaluate_dataset
from utils.plot_utils import plot_g_d_loss, plt
from utils.setup import *


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)


# Create a function to train the model
def train(G, D, dataloader, loss_mode, num_epochs, g_lr, d_lr, beta1, beta2, image_save_dir):
    print_separator()
    print('==> Training started..')

    G.to(DEVICE)
    D.to(DEVICE)

    # Set train mode
    G.train()
    D.train()

    # Define loss function
    criterion = get_loss_function(loss_mode)

    # Set up the optimization algorithm and scheduler
    g_optimizer = Adam(G.parameters(), lr=g_lr, betas=(beta1, beta2))
    d_optimizer = Adam(D.parameters(), lr=d_lr, betas=(beta1, beta2))

    g_scheduler = CosineAnnealingLR(g_optimizer, T_max=num_epochs)
    d_scheduler = CosineAnnealingLR(d_optimizer, T_max=num_epochs)

    batch_size = dataloader.batch_size

    # Set the iteration counter
    # CRITIC_ITERATIONS = 5
    G_losses = []
    D_losses = []

    for epoch in range(1, num_epochs + 1):

        # Record start time
        t_epoch_start = time()

        # Record g & d loss
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        for i, images in enumerate(dataloader):
            # --------------------
            # 1. Update D network: maximize logs(D(x)) + logs(1 - D(G(z)))
            # --------------------

            images = images.to(DEVICE)

            # Make real and fake labels: 1 & 0
            mini_batch_size = images.size()[0]
            # label_real = torch.full((mini_batch_size,), 0.96).to(DEVICE)
            label_real = torch.full((mini_batch_size,), 1.).to(DEVICE)
            label_fake = torch.full((mini_batch_size,), 0.).to(DEVICE)

            # for p in D.parameters():  # reset requires_grad
            #     p.requires_grad = True  # they are set to False below in netG update

            # critic training schedule
            # for d_iter in range(CRITIC_ITERATIONS):
            # Discriminate real images
            d_out_real = D(images)

            # Make and Discriminate fake images
            input_z = torch.randn(mini_batch_size, args.z_dim).to(DEVICE)
            if args.model_name != 'gan':
                input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_images = G(input_z)
            d_out_fake = D(fake_images)

            # Compute loss of real and fake batches as D loss
            if args.loss_mode != 'hinge':
                d_loss_real = criterion(d_out_real, label_real)
                d_loss_fake = criterion(d_out_fake, label_fake)
            if args.loss_mode == 'hinge':
                d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
                d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()

            d_loss = d_loss_real + d_loss_fake

            epoch_d_loss += d_loss.item()

            # Update D
            d_optimizer.zero_grad()
            d_loss.backward(retain_graph=True)
            d_optimizer.step()

            # --------------------
            # 2. Update G network: maximize logs(D(G(z)))
            # --------------------

            # for p in D.parameters():
            #     p.requires_grad = False  # to avoid computation

            #  Discriminate fake images
            d_out_fake = D(fake_images)

            # Compute G loss
            g_loss = None
            if loss_mode in ['vanilla', 'smooth_vanilla']:
                g_loss = criterion(d_out_fake, label_real)
            elif loss_mode == 'hinge':
                g_loss = -d_out_fake.mean()
            else:
                assert "Unknown loss mode!"

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
            D_losses.append(d_loss.item())
            G_losses.append(g_loss.item())

        t_epoch_finish = time()

        # Update learning rate
        g_scheduler.step()
        d_scheduler.step()

        # Plot the progress
        print_separator()
        print(
            f'epoch {epoch} || Epoch_D_Loss:{epoch_d_loss / batch_size:.4f} || Epoch_G_Loss:{epoch_g_loss / batch_size:.4f}')
        print(f'Time taken: {t_epoch_finish - t_epoch_start:.3f} sec.')

        # If at save interval => save generated image samples and checkpoints
        if epoch % args.save_interval == 0 or epoch > num_epochs - 20:
            # Save model
            ckpt_path = os.path.join(ckpt_save_dir, f'epoch{epoch}.pth')
            torch.save(G.state_dict(), ckpt_path)

            # Test generator
            fig = plt.figure(figsize=(5, 5))
            fig.suptitle(f"epoch {epoch}", fontsize=17)
            with torch.no_grad():
                input_z = torch.randn(batch_size // 2, args.z_dim).to(DEVICE)

                if args.model_name != 'gan':
                    input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
                fake = G(input_z).detach().cpu()
                for i in range(0, 25):
                    plt.subplot(5, 5, i + 1)
                    plt.imshow(fake[i][0].cpu().detach().numpy(), 'gray')
                    plt.axis('off')
                plt.show()

            # num_fake_images = 2534
            num_fake_normal_images = 1349
            # num_fake_pneumonia_images = 2505
            with torch.no_grad():
                # Create random tensors for computing fid value
                fixed_z = torch.randn(num_fake_normal_images, args.z_dim)
                if args.model_name != 'gan':
                    fixed_z = fixed_z.view(fixed_z.size(0), fixed_z.size(1), 1, 1)
                fake_images_ = G(fixed_z.to(DEVICE))
                print("Generating a batch of images...")

                image_save_path = os.path.join(image_save_dir, 'epoch' + str(epoch))
                create_folder(image_save_path)

                idx = 0
                for i in range(num_fake_normal_images):
                    fake_image_numpy, fake_image_ = decode_to_numpy(fake_images_[i][0])
                    path = os.path.join(image_save_path, f'fake_img{idx}.png')
                    cv.imwrite(path, fake_image_)
                    idx += 1

                print("Starting computing fid value...")
                # real_images_folder = 'data/real_valid_pneumonia_images'
                real_images_folder = 'data/real_valid_normal_images'
                generated_images_folder = image_save_path

                fid_value = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder],
                                                                batch_size=256, device='cuda', dims=2048)
                avg_psnr, avg_ssim = evaluate_dataset(real_images_folder, generated_images_folder)
                print("\033[96mFID value: {:.2f} ||  PSNR:{:.2f}  ||  SSIM:{:.2f}\033[0m".format(
                    fid_value, avg_psnr, avg_ssim))
                log_fid_value(epoch, fid_value)

        # # tensorboard visualization
        # writer.add_scalars(main_tag='Training Loss',
        #                    tag_scalar_dict={'d_loss': epoch_d_loss / batch_size,
        #                                     'g_loss': epoch_g_loss / batch_size},
        #                    global_step=epoch)
        #
        # torch.save(G.state_dict(), os.path.join(ckpt_save_dir, f'last.pth'))

    print('==> Training ended..')

    return G, D, G_losses, D_losses


def create_hyperparameter_table(args):
    """
    Create a PrettyTable containing hyperparameters and data information.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    Returns:
        PrettyTable: A table containing hyperparameters and data information.
    """
    table = PrettyTable(['Hyper-Parameters & Data Infos', 'Value'])
    table.align['Hyper-Parameters & Data Infos'] = 'l'
    table.align['Value'] = 'r'

    table.add_row(['Batch Size', args.batch_size])
    table.add_row(['Num Workers', args.num_workers])
    table.add_row(['Num Epochs', args.num_epochs])
    table.add_row(['Generator LR', args.g_lr])
    table.add_row(['Discriminator LR', args.d_lr])
    table.add_row(['beta1', args.beta1])
    table.add_row(['beta2', args.beta2])
    table.add_row(['Random Seed', args.seed])
    table.add_row(['Device', DEVICE])
    table.add_row(['Loss Function', args.loss_mode])
    table.add_row(["", ""])

    table.add_row(['dataset root', DATA_DIR])
    table.add_row(['Class Name', CLASS_NAME])
    table.add_row(['Image Size', args.image_size])
    table.add_row(['Z_dim', args.z_dim])
    table.add_row(["", ""])

    print(table)


def get_argparse():
    # Define cmd arguments
    parser = argparse.ArgumentParser()

    # default represents dcgan setting

    # DCGAN Options
    parser.add_argument('--image-size', type=int, default=128, help='image size (default: 64)')
    parser.add_argument('--z-dim', type=int, default=100, help='z-dim of generator (default: 100)')

    # modify setting
    # g-lr 0.0001   d-lr 0.0004
    # beta1 0.   beta2 0.999

    # Train Options
    parser.add_argument('--batch-size', type=int, default=64, help='batch size for training (default: 128)')
    parser.add_argument('--num-workers', type=int, default=0, help='number of data loading workers')
    parser.add_argument('--num_epochs', type=int, default=400, help='number of epochs for training (default: 200)')
    parser.add_argument('--g-lr', type=float, default=0.0001, help='learning rate of generator (default: 0.0002)')
    parser.add_argument('--d-lr', type=float, default=0.0004, help='learning rate of discriminator (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0., help='beta of generator (default: 0.5)')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta of discriminator (default: 0.999)')
    parser.add_argument('--seed', type=int, default=47, help='random seed 1234 | 47 | 3407')
    parser.add_argument('--use-gpu', action='store_true', default=True, help='turn on flag to use GPU')
    parser.add_argument('--loss-mode', type=str, default='vanilla',
                        help='gan loss for training. [vanilla | smooth_vanilla | hinge ]')

    # Model Options
    parser.add_argument('--model_name', type=str, default='dcgan', help='DCGAN [gan | dcgan | sndcgan | resdcgan')

    # File Management Options
    parser.add_argument('--save-interval', type=int, default=10, help='interval of saving checkpoints')
    parser.add_argument('--exp-name', type=str, default='exp_dcgan_base',
                        help='exp name for checkpoints directory')

    return parser


if __name__ == "__main__":
    args = get_argparse().parse_args()

    # Determine device
    DEVICE = 'cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu'

    # Define directory paths
    CONFIG_PATH = './configs/config.yaml'
    IMAGE_SAVE_DIR = f'outs/dcgan/{args.exp_name}'

    # Load YAML configuration
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        yaml_info = yaml.load(f.read(), Loader=yaml.FullLoader)
    n_channels = yaml_info['n_channels']
    DATA_DIR = yaml_info['root1']
    CLASS_NAME = yaml_info['class_name']
    MEAN = yaml_info['mean']
    STD = yaml_info['std']

    # Create and configure PrettyTable
    create_hyperparameter_table(args)

    # Set checkpoints dir
    ckpt_save_dir = os.path.join('./checkpoints/dcgan', args.exp_name).replace('\\', '/')
    create_folder(ckpt_save_dir)
    print(f"Checkpoints results to {ckpt_save_dir}.")

    # Set logs dir
    log_save_path = os.path.join('logs/dcgan/', args.exp_name).replace('\\', '/')
    create_folder(log_save_path)

    # Initialize
    writer = SummaryWriter(log_dir=log_save_path)

    # Set log info
    setup_logging(log_filename=os.path.join(log_save_path, args.exp_name).replace('\\', '/'))
    log_hyperparameters(args)

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Get dataloader
    train_dataloader = get_dataloader_dcgan(DATA_DIR, CLASS_NAME, args.batch_size, args.num_workers)

    # Initialize model and weight
    G, D = create_gan_model(args.model_name, z_dim=args.z_dim, image_size=args.image_size)

    G.apply(weights_init)
    D.apply(weights_init)

    # Train model
    G_update, D_update, G_loss_set, D_loss_set = train(G, D, train_dataloader, args.loss_mode, args.num_epochs,
                                                       args.g_lr,
                                                       args.d_lr,
                                                       args.beta1,
                                                       args.beta2,
                                                       IMAGE_SAVE_DIR)

    # Training visualization
    plot_g_d_loss(G_loss_set, D_loss_set, ckpt_save_dir)

# IMPORT PACKAGES
from prettytable import PrettyTable
from tqdm import tqdm
import argparse
import yaml
import os

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn

from utils.common_utils import create_folder, print_separator
from utils.classifier_dataset import PneumoniaOriginalDataset, PneumoniaBalancedDataset
from utils.plot_utils import plot_accuracy, plot_loss
from utils.setup import *

from models.create_model import create_cnn_model

# from onnx.export_onnx import export_onnx
# from compression.prune import main_prune

import warnings

warnings.filterwarnings("ignore")

# result reproduction
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.cuda.empty_cache()


def get_original_dataloader(batch_size, num_workers, image_size, mean, std):
    print('==> Getting original dataloader..')

    train_transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(degrees=10),

        # transforms.RandomCrop(image_size),
        # transforms.ColorJitter(contrast=0.5),
        transforms.Resize([image_size, image_size]),

        transforms.RandomResizedCrop(size=image_size, scale=(0.6, 1.2)),
        # transforms.RandomHorizontalFlip(),
        # transforms.Resize([image_size, image_size]),
        transforms.ToTensor(),
        # transforms.Normalize(mean=mean, std=std)
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize([image_size, image_size]),
        transforms.ToTensor(),
        # transforms.Normalize(mean=mean, std=std)
    ])

    train_dataset = PneumoniaOriginalDataset(real_dataset_root, mode='train', transform=train_transform)
    val_dataset = PneumoniaOriginalDataset(real_dataset_root, mode='val', transform=val_test_transform)
    test_dataset = PneumoniaOriginalDataset(real_dataset_root, mode='test', transform=val_test_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=2 * batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=4 * batch_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, val_dataloader, test_dataloader


def get_balanced_augmented_dataloader(batch_size, num_workers, image_size, mean, std):
    print('==> Getting balanced dataloader..')

    train_transform = transforms.Compose([
        transforms.Resize([image_size, image_size]),
        # RandomGaussianNoise(p=0.5, mean=0, std=1),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(degrees=10),
        # transforms.RandomCrop(image_size),
        # transforms.ColorJitter(brightness=0.2,
        #                        contrast=0.2,
        #                        saturation=0.2,
        #                        hue=0.1),
        # transforms.ColorJitter(contrast=0.7),
        transforms.RandomResizedCrop(size=image_size, scale=(0.6, 1.2)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=mean, std=std)
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize([image_size, image_size]),
        transforms.ToTensor(),
        # transforms.Normalize(mean=mean, std=std)
    ])

    train_dataset = PneumoniaBalancedDataset(real_dataset_root, fake_dataset_root, mode='train',
                                             transform=train_transform)
    val_dataset = PneumoniaBalancedDataset(real_dataset_root, fake_dataset_root, mode='val',
                                           transform=val_test_transform)
    test_dataset = PneumoniaBalancedDataset(real_dataset_root, fake_dataset_root, mode='test',
                                            transform=val_test_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=2 * batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=4 * batch_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, val_dataloader, test_dataloader


def test(model, dataloader, device):
    model.eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            _, pred = torch.max(outputs, 1)
            correct += (pred == targets).sum().item()
            total += targets.size(0)
    return correct / total


def train(model, train_dataloader, val_dataloader, test_dataloader, device, batch_size, num_epoch, lr,
          optim_policy='sgd',
          weight_decay=0, scheduler_type='cos', half_precision=False):
    print_separator()
    print('==> Training started..')

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Set up the optimization algorithm and scheduler
    if optim_policy == 'sgd':
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optim_policy == 'adam':
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler_type == 'cos':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch)
    elif scheduler_type == 'step':
        scheduler = StepLR(optimizer, args.step_size, gamma=args.gamma,
                           last_epoch=-1)

    if half_precision:
        scaler = GradScaler()
    else:
        scaler = None

    train_acc_all = []
    val_acc_all = []
    train_loss_all = []
    val_loss_all = []

    best_val_acc = 0.0
    early_stop_counter = 0
    patience = 20

    for epoch in range(1, num_epoch + 1):
        model.train()
        total_loss = 0
        loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), postfix=dict, mininterval=0.3)
        for i, (inputs, targets) in loop:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            if half_precision:
                with autocast():
                    outputs = model(inputs)

                    # Compute loss
                    loss = criterion(outputs, targets)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            if half_precision:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            # Add epoch, loss, lr
            loop.set_description(f"Epoch [{epoch}/{num_epoch}]")
            loop.set_postfix({'total_loss': total_loss / (i + batch_size),
                              'lr': scheduler.get_last_lr()})
            loop.update(2)

        scheduler.step()

        # evaluate
        train_acc = test(model, train_dataloader, device)
        val_acc = test(model, val_dataloader, device)
        test_acc = test(model, test_dataloader, device)

        # Write down accuracy
        train_acc_all.append(train_acc)
        val_acc_all.append(val_acc)

        # compute val loss
        with torch.no_grad():
            val_total_loss = 0.0
            for inputs, targets in val_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)

                # Compute loss
                val_loss = criterion(outputs, targets)

                val_total_loss += val_loss.item()

        train_loss_all.append(total_loss / len(train_dataloader))
        val_loss_all.append(val_total_loss / len(val_dataloader))

        # Early stopping & Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(ckpt_save_path, 'best.pth').replace('\\', '/'))

            print_separator()
            print(
                f"Save best model to checkpoints file!\t Train Acc: {train_acc * 100.:.2f}% Val Acc: {val_acc * 100.:.2f}% Test Acc: {test_acc * 100.:.2f}%")
            print_separator()
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f'==> Early stopping after {epoch} epochs...')
            break

    print('==> Training ended...')

    return train_acc_all, val_acc_all, train_loss_all, val_loss_all


def create_hyperparameter_table(args):
    """
    Create a PrettyTable containing hyperparameters and data information.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    Returns:
        PrettyTable: A table containing hyperparameters and data information.
    """
    table = PrettyTable(['Hyper-Parameters & data infos', 'Value'])
    table.align['Hyper-Parameters & data infos'] = 'l'
    table.align['Value'] = 'r'

    # Add to table
    table.add_row(['Batch size', args.batch_size])
    table.add_row(['NUM Workers', args.num_workers])
    table.add_row(['Num Epoch', args.num_epoch])
    table.add_row(['Device', DEVICE])
    table.add_row(['Optimizer Strategy', args.optim_policy])
    table.add_row(['Learning Rate', args.lr])
    table.add_row(['Weight Decay', args.weight_decay])
    table.add_row(['Scheduler Type', args.scheduler_type])
    table.add_row(['gamma', args.gamma])
    table.add_row(['step-size', args.step_size])
    table.add_row(['random seed', args.seed])

    table.add_row(["", ""])
    table.add_row(['use_balanced', args.use_balanced])
    table.add_row(['real_dataset_root', real_dataset_root])
    table.add_row(['fake_dataset_root', fake_dataset_root])
    table.add_row(['n_channels', n_channels])
    table.add_row(['n_classes', n_classes])
    table.add_row(['image_size', IMAGE_SIZE])
    table.add_row(['mean', MEAN])
    table.add_row(['std', STD])
    print(table)


def get_argparse():
    # Define cmd arguments
    parser = argparse.ArgumentParser()

    # Train Options
    parser.add_argument('--batch-size', type=int, default=128, help='batch size for training (default: 64)')
    parser.add_argument('--num-workers', type=int, default=0, help='number of data loading workers')
    parser.add_argument('--num-epoch', type=int, default=50, help='number of epochs for training (default: 40)')
    parser.add_argument('--optim-policy', type=str, default='adam', help='optimizer for training. [sgd | adam]')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=1e-6, help='weight-decay for training. [le-4 | 1e-6]')
    parser.add_argument('--half-precision', action='store_true', default=True,
                        help='whether to use half precision training')

    # Learning Rate Options
    parser.add_argument('--scheduler-type', type=str, default='cos', help='learning rate decay policy. [step | cos]')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='multiply by a gamma every lr_decay_epochs (only for step lr policy)')
    parser.add_argument('--step-size', type=int, default=30, help='period of lr decay. (default: 20)')

    parser.add_argument('--seed', type=int, default=3407, help='random seed. [47 | 3407 | 1234]')
    parser.add_argument('--use-gpu', action='store_true', default=True, help='turn on flag to use GPU')
    # parser.add_argument('--pretrain', action='store_true', default=True, help='whether to fine-tune')

    # Model Options
    parser.add_argument('--model_name', type=str, default='resnet18',
                        help='CNN [resnet18-sam | resnet18 | resnet34 | vgg16 | vgg19 | mobilenetv2 | alexnet]')
    parser.add_argument('--use-pretrained', action='store_true', default=False,
                        help='whether to fine-tune on pretrained model')

    # Dataset Options
    parser.add_argument('--use-balanced', action='store_true', default=True,
                        help='whether to choose balanced dataset')

    # File Management Options
    parser.add_argument('--exp-name', type=str, default='balanced_resnet18',
                        help='exp name for training')

    '''
    # Prune Options
    parser.add_argument('--baseline', type=str,
                        default='checkpoints/cls/exp_fake_scratch_pretrained_test/pneumonia_best.pth',
                        help='base model for pruning')
    parser.add_argument('--prune', action='store_true', default=False, help='start pruning')
    parser.add_argument('--prune-policy', type=str, default='fpgm', help='pruning policy. [l1 | l2 | fpgm] ')
    parser.add_argument('--sparse', type=float, default=0.8, help='pruning ratio (default: 0.8)')
    '''

    return parser


if __name__ == '__main__':
    args = get_argparse().parse_args()

    # Determine device
    DEVICE = 'cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu'

    # Define directory paths
    CONFIG_PATH = './configs/config.yaml'

    # Load YAML configuration
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        yaml_info = yaml.load(f.read(), Loader=yaml.FullLoader)
    real_dataset_root = yaml_info['root1']
    fake_dataset_root = yaml_info['root2']
    n_channels = yaml_info['n_channels']
    n_classes = yaml_info['n_classes']
    IMAGE_SIZE = yaml_info['cnn_image_size']
    MEAN = yaml_info['mean']
    STD = yaml_info['std']

    EXP_NAME = f'lr{args.lr}_{args.optim_policy}_seed{args.seed}'

    # Create and configure PrettyTable
    create_hyperparameter_table(args=args)

    # Create directory for saving model
    ckpt_save_path = os.path.join('checkpoints/cnn', args.exp_name, EXP_NAME).replace('\\', '/')
    create_folder(ckpt_save_path)
    print(f"Checkpoints results to {ckpt_save_path}.")

    # Set logs dir
    log_save_path = os.path.join('logs/cnn', args.exp_name).replace('\\', '/')
    create_folder(log_save_path)

    # Set config log
    setup_logging(log_filename=os.path.join(log_save_path, EXP_NAME).replace('\\', '/'))
    log_hyperparameters(args=args)

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Get dataloader
    if args.use_balanced:
        train_dataloader, val_dataloader, test_dataloader = get_balanced_augmented_dataloader(args.batch_size,
                                                                                              args.num_workers,
                                                                                              IMAGE_SIZE, MEAN, STD)
    else:
        train_dataloader, val_dataloader, test_dataloader = get_original_dataloader(args.batch_size, args.num_workers,
                                                                                    IMAGE_SIZE, MEAN, STD)

    # Initialize model and weight
    model = create_cnn_model(args.model_name, args.use_pretrained)

    # Start Pruning
    from nni.compression.pytorch.pruning import FPGMPruner
    from nni.compression.pytorch import ModelSpeedup
    from nni.compression.pytorch.utils import count_flops_params

    # ratio = 0.6
    # config_list = [{'sparsity': ratio,
    #                 'op_types': ['Conv2d']}]
    #
    # model = create_cnn_model('resnet18', pretrained=False)
    # state_dict = torch.load('compression/resnet18_baseline.pth')
    # model.load_state_dict(state_dict)

    # dummy_input = torch.rand(1, 1, 224, 224).to('cuda')
    #
    # flops2, params2, _ = count_flops_params(model, dummy_input, verbose=True)
    # print(f"\nModel:\nFLOPs {flops2 / 1e6:.2f}M, Params {params2 / 1e6:.2f}M")
    #
    # pruner = FPGMPruner(model, config_list)
    # _, masks = pruner.compress()
    # pruner._unwrap_model()
    #
    # ModelSpeedup(model, dummy_input, masks).speedup_model()
    #
    # flops2, params2, _ = count_flops_params(model, dummy_input, verbose=True)
    # print(f"\nPruned Model:\nFLOPs {flops2 / 1e6:.2f}M, Params {params2 / 1e6:.2f}M")
    #

    # Train model
    train_acc_set, val_acc_set, train_loss_set, val_loss_set = train(model,
                                                                     train_dataloader,
                                                                     val_dataloader,
                                                                     test_dataloader,
                                                                     DEVICE,
                                                                     args.batch_size,
                                                                     args.num_epoch, args.lr,
                                                                     args.optim_policy,
                                                                     args.weight_decay,
                                                                     args.scheduler_type,
                                                                     args.half_precision)

    # # export onnx
    # torch.onnx.export(model, dummy_input, 'onnx/resnet18-lite-0.6.onnx', input_names=['input'],
    #                   output_names=['output'], verbose=True,
    #                   opset_version=11)
    #
    # print("Successfully exporting onnx model!")

    plot_accuracy(train_acc_set, val_acc_set, save_dir=log_save_path)
    plot_loss(train_loss_set, val_loss_set, save_dir=log_save_path)

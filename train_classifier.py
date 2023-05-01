# IMPORT PACKAGES
from A import PrettyTable
from tqdm import tqdm
import numpy as np
import argparse
import random
import yaml

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch

from utils.classifier_dataset import PneumoniaDataset, ImageTransform
from utils.common_utils import create_folder, print_info
from utils.plot_utils import plot_accuracy

from models.create_model import create_model

# from onnx.export_onnx import export_onnx
# from compression.prune import main_prune

# result reproduction
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.cuda.empty_cache()


def set_seed(seed):
    """
    Set random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def test(split):
    """
    This function takes in a string 'split' and returns the accuracy of the model on the specified data split.
    :param split:train/val/test
    :return:accuracy
    """
    model.eval()

    data_info = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    data_loader = data_info[split]

    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)

            _, pred = torch.max(outputs, 1)
            correct += (pred == targets).sum().item()
            total += targets.size(0)

    return correct / total


def train(model, epochs):
    # Define loss function
    criterion = nn.CrossEntropyLoss()

    train_acc_all = []
    test_acc_all = []
    val_acc_all = []
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        loop = tqdm(enumerate(train_loader), total=len(train_loader), postfix=dict, mininterval=0.3)
        for i, (inputs, targets) in loop:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)

            # Update model
            optimizer.zero_grad()
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

            # Add epoch, loss, lr
            loop.set_description(f"Epoch [{epoch}/{epochs}]")
            loop.set_postfix({'total_loss': total_loss / (i + BATCH_SIZE),
                              'lr': scheduler.get_last_lr()})
            loop.update(1)

        scheduler.step()

        # evaluate
        train_acc = test('train')
        test_acc = test('test')
        val_acc = test('val')

        # Write down accuracy
        train_acc_all.append(train_acc)
        test_acc_all.append(test_acc)
        val_acc_all.append(val_acc)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'{save_dir}/best.pth')

            print_info()
            print(
                f"Save best model to checkpoints file!\t Train Acc: {best_acc * 100.:.2f}% Val Acc: {val_acc * 100.:.2f}% Test Acc: {test_acc * 100.:.2f}")
            print_info()
    return train_acc_all, test_acc_all, val_acc_all


def get_argparse():
    # Define cmd arguments
    parser = argparse.ArgumentParser()

    # Train Options
    parser.add_argument('--batch-size', type=int, default=128, help='batch size for training (default: 32)')
    parser.add_argument('--workers', type=int, default=12, help='number of data loading workers')
    parser.add_argument('--epochs', type=int, default=40, help='number of epochs for training (default: 20)')
    parser.add_argument('--optim-policy', type=str, default='adam', help='optimizer for training. [sgd | adam]')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=1e-6, help='weight-decay for training. [le-4 | 1e-6]')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd (default: 0.9)')
    parser.add_argument('--lr-policy', type=str, default='cos', help='learning rate decay policy. [step | cos]')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='multiply by a gamma every lr_decay_epochs (only for step lr policy)')
    parser.add_argument('--step-size', type=int, default=20, help='period of lr decay. (default: 20)')
    parser.add_argument('--seed', type=int, default=1234, help='random seed. [47 | 3407 | 1234]')
    parser.add_argument('--use-gpu', action='store_true', default=True, help='turn on flag to use GPU')
    parser.add_argument('--pretrain', action='store_true', default=True, help='whether to fine-tune')

    # Model Options
    parser.add_argument('--model_name', type=str, default='resnet18-sam',
                        help='CNN [resnet18-cbam | resnet18 | seresnet18 | resnet34 | vgg16 | vgg19 | mobilenetv2 | alexnet]')
    parser.add_argument('--use-pretrained', action='store_true', default=False,
                        help='whether to fine-tune on pretrained model')

    '''
    # Prune Options
    parser.add_argument('--baseline', type=str,
                        default='checkpoints/cls/exp_fake_scratch_pretrained_test/pneumonia_best.pth',
                        help='base model for pruning')
    parser.add_argument('--prune', action='store_true', default=False, help='start pruning')
    parser.add_argument('--prune-policy', type=str, default='fpgm', help='pruning policy. [l1 | l2 | fpgm] ')
    parser.add_argument('--sparse', type=float, default=0.8, help='pruning ratio (default: 0.8)')
    '''

    # File Management Options
    parser.add_argument('--name', type=str, default='exp1',
                        help='exp name for training')

    return parser


if __name__ == '__main__':
    args = get_argparse().parse_args()

    # hyper-parameters
    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    WORKERS = args.workers
    LR = args.lr
    MOMENTUM = args.momentum
    DEVICE = 'cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu'

    # data info
    with open('./configs/config.yaml', 'r', encoding='utf-8') as f:
        yaml_info = yaml.load(f.read(), Loader=yaml.FullLoader)
    dataset_root1 = yaml_info['root1']
    dataset_root2 = yaml_info['root2']
    n_channels = yaml_info['n_channels']
    n_classes = yaml_info['n_classes']
    image_size = yaml_info['image_size']

    # Set save dir
    train_infos = [args.name, 'lr', LR, args.lr_policy, args.step_size]
    log_info = "_".join(str(i) for i in train_infos)
    save_dir = f"checkpoints/cls/{log_info}"
    create_folder(save_dir)
    # print(f"save_dir={save_dir}")

    # Create the table object, name, and alignment
    table = PrettyTable(['Hyper-Parameters & data infos', 'Value'])
    table.align['Hyper-Parameters & data infos'] = 'l'
    table.align['Value'] = 'r'

    # Add to table
    table.add_row(['Batch size', BATCH_SIZE])
    table.add_row(['Workers', WORKERS])
    table.add_row(['Num epochs', NUM_EPOCHS])
    table.add_row(['Optimizer strategy', args.optim_policy])
    table.add_row(['Weight decay', args.weight_decay])
    table.add_row(['Learning rate', LR])
    table.add_row(['Momentum', MOMENTUM])
    table.add_row(['LR policy', args.lr_policy])
    table.add_row(['gamma', args.gamma])
    table.add_row(['step-size', args.step_size])
    table.add_row(['random seed', args.seed])
    table.add_row(['Device', DEVICE])
    table.add_row(["", ""])
    table.add_row(['dataset_root1', dataset_root1])
    table.add_row(['dataset_root2', dataset_root2])
    table.add_row(['n_channels', n_channels])
    table.add_row(['n_classes', n_classes])
    table.add_row(['image_size', image_size])
    print(table)

    set_seed(args.seed)

    # Get dataset and prepare dataloader
    print_info()
    print('==> Getting dataloader..')
    train_dataset = PneumoniaDataset(dataset_root1, dataset_root2, mode='train', transform=ImageTransform())
    print_info()
    val_dataset = PneumoniaDataset(dataset_root1, dataset_root2, mode='val', transform=ImageTransform())
    print_info()
    test_dataset = PneumoniaDataset(dataset_root1, dataset_root2, mode='test', transform=ImageTransform())
    print_info()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=2 * BATCH_SIZE, shuffle=False, num_workers=WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=4 * BATCH_SIZE, shuffle=False, num_workers=WORKERS)

    # Create models
    print_info()
    print('==> Building model..')
    model = create_model(args.model_name, args.use_pretrained)

    print_info()
    print('==> Defining optimizer and scheduler..')
    if args.optim_policy == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    else:
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=args.weight_decay)

    if args.lr_policy == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
    elif args.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, gamma=args.gamma, last_epoch=-1)

    print_info()
    print('==> Training model..')
    train_acc_set, test_acc_set, val_acc_set = train(model, NUM_EPOCHS)

    plot_accuracy(NUM_EPOCHS, train_acc_set, test_acc_set, val_acc_set, save_dir)

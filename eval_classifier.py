import argparse
import yaml
import os

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch

from utils.classifier_dataset import PneumoniaBalancedDataset
from utils.metrics import classifier_evaluate
from utils.plot_utils import plot_confusion_matrix
from utils.common_utils import create_folder

from models.create_model import create_cnn_model


def get_balanced_augmented_dataloader(real_dataset_root, fake_dataset_root, batch_size, num_workers, image_size,
                                      ):
    print('==> Getting Test dataloader..')

    test_transform = transforms.Compose([
        transforms.Resize([image_size, image_size]),
        transforms.ToTensor(),
    ])

    test_dataset = PneumoniaBalancedDataset(real_dataset_root, fake_dataset_root, mode='test',
                                            transform=test_transform)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return test_dataloader


def predict(model, data_loader):
    """
    Tests the model on the given data_loader and returns the predicted and actual labels.

    Args:
    - model: the model to be tested
    - data_loader: the data loader to be used for testing

    Returns:
    - pred_lst: a list of predicted labels
    - label_lst: a list of actual labels
    """
    model.eval()
    pred_lst = []
    label_lst = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)

            _, pred = torch.max(outputs, 1)
            pred_lst.extend(pred.cpu().numpy())
            label_lst.extend(targets.cpu().numpy())

    return pred_lst, label_lst


def eval(model_name, ckpt_path, test_dataloader, device, save_dir='figures/confusion_matrix/'):
    create_folder(save_dir)

    # build model
    model = create_cnn_model(model_name, pretrained=False)

    # load checkpoints
    try:
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        print(e)

    # Get pred and labels for evaluation
    pred, labels = predict(model, test_dataloader)

    # get metrics
    accuracy, precision, recall, f1, cm = classifier_evaluate(pred, labels)
    print(f"The accuracy: {100. * accuracy:.2f} %")
    print(f"The precision: {100. * precision:.2f} %")
    print(f"The recall: {100. * recall:.2f} %")
    print(f"f1: {100. * f1:.2f} %")
    print(f"Confusion matrix: {cm}")

    # save cm
    plot_confusion_matrix(cm, save_dir, title='ChestXRay-CM')


if __name__ == '__main__':
    # Define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', type=str, required=False, default='pretrained/new/resnet18-sam.pth',
                        help='checkpoints path for inference')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size for training (default: 32)')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--device', type=str, default='cuda', required=False,
                        help='inference device. [cuda | cpu]')
    parser.add_argument('--model_name', type=str, default='resnet18-sam',
                        help='CNN [resnet18-sam  resnet18 | seresnet18 | resnet34 | vgg16 | vgg19 | mobilenetv2 | ')
    args = parser.parse_args()

    # Check input arguments
    if not os.path.exists(args.ckpt_path):
        print(f'Cannot find the checkpoints: {args.ckpt_path}')
        exit()

    # Load YAML configuration
    CONFIG_PATH = './configs/config.yaml'
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        yaml_info = yaml.load(f.read(), Loader=yaml.FullLoader)
    REAL_DATASET_ROOT = yaml_info['root1']
    FAKE_DATASET_ROOT = yaml_info['root2']
    IMAGE_SIZE = yaml_info['cnn_image_size']
    MEAN = yaml_info['mean']
    STD = yaml_info['std']

    # Load test dataset
    test_dataloader = get_balanced_augmented_dataloader(REAL_DATASET_ROOT, FAKE_DATASET_ROOT, args.batch_size,
                                                        args.num_workers,
                                                        IMAGE_SIZE)

    # Run to test
    eval(args.model_name, args.ckpt_path, test_dataloader, args.device)

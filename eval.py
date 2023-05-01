import argparse
import os

import torch.utils.data as data
import torch

from utils.classifier_dataset import PneumoniaDataset, ImageTransform
from utils.metrics import classifier_evaluate
from utils.plot_utils import plot_confusion_matrix
from utils.common_utils import create_folder

from models.create_model import create_model


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


def test(ckpt_path, device, save_dir='figures/confusion_matrix/'):
    create_folder(save_dir)

    # build model
    model = create_model('resnet18-sam', pretrained=False)
    if model is None:
        assert "model can not be None!"

    # load checkpoints
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Get pred and labels for evaluation
    pred, labels = predict(model, test_loader)

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
    parser.add_argument('--ckpt-path', type=str, required=True, default='pretrained/resnet18-sam.pth',
                        help='checkpoints path for inference')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='batch size for training (default: 32)')
    parser.add_argument('--workers', type=int, default=0,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='inference device. [cuda | cpu]')
    args = parser.parse_args()

    # Check input arguments
    if not os.path.exists(args.ckpt_path):
        print(f'Cannot find the checkpoints: {args.ckpt_path}')
        exit()

    # Load test dataset
    test_dataset = PneumoniaDataset('./data/real', './data/fake', mode='test',
                                    transform=ImageTransform())
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # Run to test
    test(args.ckpt_path, args.device)

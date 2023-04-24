import torch

import torchvision.transforms as transforms
import torch.utils.data as data

from utils.classifier_dataset import PneumoniaDataset, ImageTransform
from utils.metrics import classifier_evaluate
from utils.plot_utils import plot_confusion_matrix

from models.model import create_model

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[.5], std=[.5])
])


def test(model, data_loader):
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


if __name__ == '__main__':
    ckpt_path = 'pretrained/resnet18-sam.pth'
    save_dir = '/'.join(ckpt_path.split('/')[:-1])
    batch_size = 32
    workers = 4
    device = 'cpu'

    # Build model
    model = create_model('resnet18-sam', pretrained=False)
    if model is None:
        assert "model can not be None!"

    # Load checkpoints
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Get new test dataloader
    test_dataset = PneumoniaDataset('./data/real', './data/fake', mode='test',
                                    transform=ImageTransform())
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    # Get pred and labels for evaluation
    pred, labels = test(model, test_loader)

    accuracy, precision, recall, f1, cm = classifier_evaluate(pred, labels)
    print(f"The accuracy: {100. * accuracy:.2f} %")
    print(f"The precision: {100. * precision:.2f} %")
    print(f"The recall: {100. * recall:.2f} %")
    print(f"f1: {100. * f1:.2f} %")
    print(f"Confusion matrix: {cm}")

    plot_confusion_matrix(cm, save_dir, title='ChestXRay_cm')

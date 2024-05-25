import torch
import torch.nn as nn


class OHEMLoss(nn.Module):
    def __init__(self, threshold=0.5):
        super(OHEMLoss, self).__init__()
        self.threshold = threshold

    def forward(self, input, target):
        loss = nn.CrossEntropyLoss(reduction='none')(input, target)  # Compute cross-entropy loss
        num_samples = len(loss)
        num_hard_samples = int(num_samples * self.threshold)  # Select number of hard samples based on threshold
        sorted_loss, _ = torch.sort(loss, descending=True)  # Sort loss values in descending order
        hard_samples_loss = sorted_loss[:num_hard_samples]  # Select top num_hard_samples maximum loss values
        return torch.mean(hard_samples_loss)  # Return the mean loss of hard samples


class VanillaLoss(nn.Module):
    def __init__(self):
        super(VanillaLoss, self).__init__()

    def forward(self, output, target):
        loss = nn.BCEWithLogitsLoss(reduction='mean')(output.view(-1), target.view(-1))
        # loss = nn.BCEWithLogitsLoss(reduction='mean')(output, target.view(-1))
        return loss


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, output, target):
        # Smooth the target labels
        smoothed_target = target * (1 - self.smoothing) + 0.5 * self.smoothing

        # Compute binary cross-entropy with smoothed target
        loss = nn.BCEWithLogitsLoss(reduction='mean')(output.view(-1), smoothed_target.float())
        return loss


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, output, target):
        # Compute hinge loss
        loss = torch.nn.ReLU()(1.0 - output).mean()
        return loss


class WassersteinLoss(nn.Module):
    def __init__(self):
        super(WassersteinLoss, self).__init__()

    def forward(self, real_output, fake_output):
        # Compute Wasserstein distance loss
        return torch.mean(real_output) - torch.mean(fake_output)


def get_loss_function(loss_mode):
    if loss_mode == 'vanilla':
        return VanillaLoss()
    elif loss_mode == 'smooth_vanilla':
        return LabelSmoothingLoss(smoothing=0.5)
    elif loss_mode == 'hinge':
        return HingeLoss()
    elif loss_mode == 'w_distance':
        return WassersteinLoss()
    else:
        raise ValueError("Unsupported loss mode. Please choose from 'smooth_BCE', 'BCE', 'hinge', or 'w_distance'.")


if __name__ == '__main__':
    # Example usage:
    loss_mode = 'smooth_BCE'  # Set your desired loss mode
    criterion = get_loss_function(loss_mode)

    real_label = torch.full((1,), 1.)
    fake_label = torch.full((1,), 0.)
    D_loss = criterion(real_label, fake_label)

    print(D_loss)

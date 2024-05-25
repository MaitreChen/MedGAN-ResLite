import os
import cv2

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.classifier_dataset import PneumoniaOriginalDataset

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def classifier_evaluate(pred, label):
    accuracy = accuracy_score(label, pred, normalize=True)

    # The positive and negative sample ratio is balanced,so use macro
    average_mode = 'macro'
    precision = precision_score(label, pred, average=average_mode)
    recall = recall_score(label, pred, average=average_mode)
    f1 = f1_score(label, pred, average=average_mode)
    cm = confusion_matrix(label, pred, labels=None, sample_weight=None)

    return accuracy, precision, recall, f1, cm


def calculate_psnr(image1, image2):
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)

    return peak_signal_noise_ratio(img1, img2)


def calculate_ssim(image1, image2):
    img1 = cv2.imread(image1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2, cv2.IMREAD_GRAYSCALE)

    return structural_similarity(img1, img2)


def evaluate_dataset(real_dataset_path, generated_dataset_path):
    real_images = sorted(os.listdir(real_dataset_path))
    generated_images = sorted(os.listdir(generated_dataset_path))

    psnr_total = 0.0
    ssim_total = 0.0

    for real_image, generated_image in zip(real_images, generated_images):
        real_image_path = os.path.join(real_dataset_path, real_image)
        generated_image_path = os.path.join(generated_dataset_path, generated_image)

        # compute psnr & ssim
        psnr = calculate_psnr(real_image_path, generated_image_path)
        ssim = calculate_ssim(real_image_path, generated_image_path)

        # accumulate value
        psnr_total += psnr
        ssim_total += ssim

    num_images = len(real_images)
    avg_psnr = psnr_total / num_images
    avg_ssim = ssim_total / num_images

    return avg_psnr, avg_ssim


def calculate_fid(real_images_folder, generated_images_folder):
    from pytorch_fid import fid_score
    fid_value = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder],
                                                    batch_size=128, device='cuda', dims=2048)

    return fid_value


def calculate_mean_std(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    mean = 0.0
    std = 0.0
    total_images = 0

    # compute mean and std
    for images, _ in dataloader:
        batch_size = images.size(0)
        total_images += batch_size
        images = images.view(batch_size, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= total_images
    std /= total_images

    return mean, std


if __name__ == "__main__":
    # real_img = '../data/real/train/normal/IM-0115-0001.jpeg'
    # fake_img = '../data/fake3/train/normal/fake_img0.png'
    # print(f"psnr:{calculate_psnr(real_img, fake_img)} ssim:{calculate_ssim(real_img, fake_img)}")

    # real_dataset_path = '../data/real_valid_normal_images'
    # real_dataset_path = '../data/real-test/test/normal'
    # real_dataset_path = '../data/real-test/test/normal'
    # generated_dataset_path = '../outs/dcgan/exp5_normal_sndcgan_cosin_hinge/epoch400'

    # print(calculate_fid(real_dataset_path, generated_dataset_path))

    # # run eval func
    # avg_psnr, avg_ssim = evaluate_dataset(real_dataset_path, generated_dataset_path)
    # print(f"Average PSNR: {avg_psnr}")
    # print(f"Average SSIM: {avg_ssim}")

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = PneumoniaOriginalDataset('../data/real/', mode='train', transform=transform)

    mean, std = calculate_mean_std(train_dataset)
    print("Mean:", mean)
    print("Standard Deviation:", std)

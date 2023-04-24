import torch
import torch.nn as nn

from models.spectral import SpectralNorm


class Generator(nn.Module):

    def __init__(self, z_dim=100, image_size=128, out_channels=3):
        super(Generator, self).__init__()

        self.layer1 = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(z_dim, image_size * 8,
                                            kernel_size=4, stride=1)),
            nn.BatchNorm2d(image_size * 8),
            nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(image_size * 8, image_size * 4,
                                            kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(image_size * 4),
            nn.ReLU(inplace=True))

        self.layer3 = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(image_size * 4, image_size * 2,
                                            kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(image_size * 2),
            nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(image_size * 2, image_size,
                                            kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(image_size),
            nn.ReLU(inplace=True))

        self.last = nn.Sequential(
            nn.ConvTranspose2d(image_size, out_channels, kernel_size=4,
                               stride=2, padding=1),
            nn.Tanh())

    def forward(self, z):
        out = self.layer1(z)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.last(out)

        return out


class Discriminator(nn.Module):

    def __init__(self, image_size=64, in_channels=3):
        super(Discriminator, self).__init__()

        self.layer1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels, image_size, kernel_size=4,
                                   stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True))

        self.layer2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(image_size, image_size * 2, kernel_size=4,
                                   stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True))

        self.layer3 = nn.Sequential(
            SpectralNorm(nn.Conv2d(image_size * 2, image_size * 4, kernel_size=4,
                                   stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True))

        self.layer4 = nn.Sequential(
            SpectralNorm(nn.Conv2d(image_size * 4, image_size * 8, kernel_size=4,
                                   stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True))

        self.last = nn.Conv2d(image_size * 8, 1, kernel_size=4, stride=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.last(out)

        return out


if __name__ == "__main__":
    G = Generator(z_dim=100, image_size=64, out_channels=1)
    D = Discriminator(image_size=64, in_channels=1)

    # Generate fake image
    input_z = torch.randn(1, 100)
    input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
    print(input_z.shape)
    fake_images = G(input_z)

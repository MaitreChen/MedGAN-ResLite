import torch
import torch.nn as nn


# from models.attention import Self_Attn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0.2)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):

    def __init__(self, z_dim=100, image_size=128):
        super(Generator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(z_dim, image_size * 8,
                               kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(image_size * 8),
            nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(image_size * 8, image_size * 4,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(image_size * 4),
            nn.ReLU(inplace=True))

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(image_size * 4, image_size * 2,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(image_size * 2),
            nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(image_size * 2, image_size,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(image_size),
            nn.ReLU(inplace=True))

        self.last = nn.Sequential(
            nn.ConvTranspose2d(image_size, 1, kernel_size=4,
                               stride=2, padding=1),
            nn.Tanh())

        # self.apply(weights_init)

        # self.attn1 = Self_Attn(128, 'relu')
        # self.attn2 = Self_Attn(64, 'relu')

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x, p1 = self.attn1(x)
        x = self.layer4(x)
        # x, p2 = self.attn2(x)
        x = self.last(x)

        return x


class Discriminator(nn.Module):

    def __init__(self, image_size=128):
        super(Discriminator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, image_size, kernel_size=4,
                      stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True))

        self.layer2 = nn.Sequential(
            nn.Conv2d(image_size, image_size * 2, kernel_size=4,
                      stride=2, padding=1),
            nn.BatchNorm2d(image_size * 2),
            nn.LeakyReLU(0.2, inplace=True))

        self.layer3 = nn.Sequential(
            nn.Conv2d(image_size * 2, image_size * 4, kernel_size=4,
                      stride=2, padding=1),
            nn.BatchNorm2d(image_size * 4),
            nn.LeakyReLU(0.2, inplace=True))

        self.layer4 = nn.Sequential(
            nn.Conv2d(image_size * 4, image_size * 8, kernel_size=4,
                      stride=2, padding=1),
            nn.BatchNorm2d(image_size * 8),
            nn.LeakyReLU(0.2, inplace=True))

        self.last = nn.Sequential(
            nn.Conv2d(image_size * 8, 1, kernel_size=4,
                      stride=1, padding=0),
            # nn.Sigmoid()
        )

        # self.apply(weights_init)

        # self.attn2 = Self_Attn(512, 'relu')

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x, p1 = self.attn1(x)
        x = self.layer4(x)
        # x, p2 = self.attn2(x)
        x = self.last(x)

        return x


if __name__ == "__main__":
    G = Generator(z_dim=100, image_size=64)
    D = Discriminator(image_size=64)

    # Generate fake image
    input_z = torch.randn(1, 100)
    input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
    print(input_z.shape)
    fake_images = G(input_z)
    print(fake_images.shape)

    # Give into discriminator
    print("")
    d_out = D(fake_images)
    print(nn.Sigmoid()(d_out))

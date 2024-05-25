import math

from torchvision.models import resnet18
import torch.nn as nn
import torch

from models.attention import ChannelAttention, SpatialAttention
from nni.compression.pytorch.utils import count_flops_params


class MedResNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(MedResNet, self).__init__()

        self.model = resnet18(pretrained=True)

        #  freeze
        for params in self.model.parameters():
            params.requires_grad_ = False

        # modify input channel
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # modify classes
        n_filters = self.model.fc.in_features
        self.model.fc = nn.Linear(n_filters, num_classes)

        # self.model.fc = nn.Conv2d(n_filters, num_classes, kernel_size=1, stride=1, bias=False) # fc2conv

    def forward(self, x):
        x = self.model(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel
                               , kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channel)

        # self.ca = ChannelAttention(out_channel)
        self.sa = SpatialAttention()

        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x) #1 64 56 56 --> 1 64 56 56
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.sa(out) * out
        # out = self.ca(out) * out

        out = self.conv2(out) # 1 64 56 56
        out = self.bn2(out)

        out = self.sa(out) * out
        # out = self.ca(out) * out

        out += identity # 1 64 56 56
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channel
        self.bn1 = nn.BatchNorm2d(out_channel)
        # --------------------------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # --------------------------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity

        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, blocks_num, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channel = 64

        # ----------------7x7 conv --------------------
        self.conv1 = nn.Conv2d(1, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        # self.conv1 = nn.Conv2d(1, self.in_channel, kernel_size=101, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)

        # ----------------3x3 max pool ----------------
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ------------------conv2_x--------------------
        self.layer1 = self._make_layer(block, 64, blocks_num[0])

        # ------------------conv3_x--------------------
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)

        # ------------------conv4_x--------------------
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)

        # ------------------conv5_x--------------------
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        # ------------------average pool and fc--------
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # output size=(1,1)

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    net = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=2).cuda()
    dummy_input = torch.randn(1, 1, 224, 224).cuda()

    print(net)

    # net(dummy_input)
    # Calculate the parameters and computational complexity of the pruned model
    flops, params, _ = count_flops_params(net, dummy_input, verbose=True)
    print(f"\nPruned Model after Weight Replacing:\nFLOPs {flops / 1e6:.2f}M, Params {params / 1e6:.2f}M")

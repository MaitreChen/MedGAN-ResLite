import torch.utils.model_zoo as model_zoo
import torch

from models.alexnet import AlexNet
from models.vgg import VGG
from models.mobilenetv2 import MobileNetV2
from models.resnet import ResNet, BasicBlock, Bottleneck, MedResNet
from models.model_urls import vgg_model_urls, resnet_model_urls

from utils.common_utils import print_separator


def resnet18_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=2).cuda()
    if pretrained:
        # pretrained_state_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth')
        pretrained_state_dict = torch.load('./pretrained/new_resnet.pth')
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict, strict=False)

    return model


def resnet18_se():
    return SEResNet(BasicResidualSEBlock, [2, 2, 2, 2]).cuda()


def resnet(pretrained=False, depth=18, num_classes=2, **kwargs):
    """Constructs a Resnet model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = None
    if depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    elif depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    elif depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    elif depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)
    model = model.cuda()

    if pretrained:
        pretrained_state_dict = None
        if depth == 18:
            pretrained_state_dict = model_zoo.load_url(resnet_model_urls['resnet18'])
        elif depth == 34:
            pretrained_state_dict = model_zoo.load_url(resnet_model_urls['resnet34'])
        elif depth == 50:
            pretrained_state_dict = model_zoo.load_url(resnet_model_urls['resnet50'])

        pretrained_state_dict.update(pretrained_state_dict)
        model.load_state_dict(pretrained_state_dict)

    return model


def alexnet(pretrained=False, **kwargs):
    """Constructs a AlexNet model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(num_classes=2).cuda()
    if pretrained:
        pretrained_state_dict = model_zoo.load_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
        pretrained_state_dict.update(pretrained_state_dict)
        model.load_state_dict(pretrained_state_dict)

    return model


def vgg(pretrained=False, depth=16, **kwargs):
    """Constructs a VGG model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        :param depth: layers of vgg net, such as vgg-16、vgg-19
    """
    model = VGG(depth=depth, num_classes=2).cuda()
    if pretrained:

        pretrained_state_dict = None
        if depth == 16:
            pretrained_state_dict = model_zoo.load_url(vgg_model_urls['vgg16'])
        elif depth == 19:
            pretrained_state_dict = model_zoo.load_url(vgg_model_urls['vgg19'])

        pretrained_state_dict.update(pretrained_state_dict)
        model.load_state_dict(pretrained_state_dict)

    return model


def mobilenetv2(num_classes=2, **kwargs):
    """
    Constructs a MobileNet V2 model
    """
    model = MobileNetV2(num_classes=num_classes).cuda()
    return


def densenet(pretrained=False, depth=121, num_classes=2, **kwargs):
    """Constructs a DenseNet model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    from models.densenet import DenseNet, Bottleneck
    model = None
    if depth == 121:
        model = DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=32, num_class=num_classes)
    elif depth == 169:
        model = DenseNet(Bottleneck, [6, 12, 32, 32], growth_rate=32, num_class=num_classes)
    elif depth == 201:
        model = DenseNet(Bottleneck, [6, 12, 48, 32], growth_rate=32, num_class=num_classes)
    elif depth == 161:
        model = DenseNet(Bottleneck, [6, 12, 36, 24], growth_rate=48, num_class=num_classes)
    model = model.cuda()

    if pretrained:
        pretrained_state_dict = model_zoo.load_url(vgg_model_urls['vgg16'])
        pretrained_state_dict.update(pretrained_state_dict)
        model.load_state_dict(pretrained_state_dict)

    return model


def create_cnn_model(model_name, pretrained=False):
    print_separator()
    print('==> Building model..')

    model = None
    if model_name == 'resnet18-sam':
        model = resnet18_cbam(pretrained)
    elif model_name == 'resnet-se':
        model = resnet18_se()
    elif model_name == 'resnet18':
        model = resnet(pretrained, depth=18)
    elif model_name == 'resnet34':
        model = resnet(pretrained, depth=34)
    elif model_name == 'resnet50':
        model = resnet(pretrained, depth=50)
    elif model_name == 'alexnet':
        model = alexnet(pretrained)
    elif model_name == 'vgg16':
        model = vgg(pretrained, depth=16)
    elif model_name == 'vgg19':
        model = vgg(pretrained, depth=19)
    elif model_name == 'mobilenetv2':
        model = MobileNetV2(num_classes=2).cuda()
    elif model_name == 'medresnet':
        model = MedResNet(in_channels=1, num_classes=2).cuda()
    else:
        raise ValueError("Unsupported CNN Model. Please check your model!")

    return model


def create_gan_model(model_name, z_dim, image_size):
    print_separator()
    print('==> Building model..')

    G = None
    D = None

    if model_name == 'gan':
        from models.gan import Generator, Discriminator
        G = Generator()
        D = Discriminator()
    elif model_name == 'dcgan':
        from models.dcgan import Generator, Discriminator
        G = Generator(z_dim, image_size)
        D = Discriminator(image_size)
    elif model_name == 'sndcgan':
        from models.sndcgan import Generator, Discriminator
        G = Generator(z_dim, image_size)
        D = Discriminator(image_size)
    elif model_name == 'wgan' or model_name == 'wgan-gp':
        from models.wgan import Generator, Discriminator
        G = Generator(z_dim, image_size)
        D = Discriminator(image_size)
    elif model_name == 'resdcgan':
        from models.resdcgan import Generator, Discriminator
        G = Generator(z_dim, image_size)
        D = Discriminator(image_size)
    else:
        raise ValueError("Unsupported GAN Model!")

    return G, D


if __name__ == '__main__':
    name = 'resnet18-sam'
    net = create_cnn_model(name, True).cuda()
    dummy_input = torch.randn(1, 1, 224, 224).cuda()

    '''
    # Calculate the parameters and computational complexity of the pruned model
    from nni.compression.pytorch.utils import count_flops_params

    flops, params, _ = count_flops_params(net, dummy_input, verbose=False)
    print(f"\nPruned Model after Weight Replacing:\nFLOPs {flops / 1e6:.2f}M, Params {params / 1e6:.2f}M")
    '''

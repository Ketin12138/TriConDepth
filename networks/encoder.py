import torch.nn as nn
from torchvision.models import resnet50, resnet101, densenet161

class ResNetBackbone(nn.Module):
    def __init__(self, depth='resnet101', pretrained=True):
        super().__init__()
        if depth == 'resnet50':
            resnet = resnet50(pretrained=pretrained)
        elif depth == 'resnet101':
            resnet = resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported ResNet depth: {depth}")

        self.stage1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
        self.stage2 = resnet.layer2
        self.stage3 = resnet.layer3
        self.stage4 = resnet.layer4

    def forward(self, x):
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f1, f2, f3, f4]

class DenseNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        densenet = densenet161(pretrained=pretrained)
        features = densenet.features
        self.stage1 = nn.Sequential(features.conv0, features.norm0, features.relu0, features.pool0, features.denseblock1)
        self.stage2 = nn.Sequential(features.transition1, features.denseblock2)
        self.stage3 = nn.Sequential(features.transition2, features.denseblock3)
        self.stage4 = nn.Sequential(features.transition3, features.denseblock4)

    def forward(self, x):
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f1, f2, f3, f4]

def get_backbone(name='resnet101', pretrained=True):
    if name == 'resnet50' or name == 'resnet101':
        return ResNetBackbone(depth=name, pretrained=pretrained)
    elif name == 'densenet161':
        return DenseNetBackbone(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown backbone name: {name}")
import torch
import torch.nn as nn
from gen import Generator
from gaussian_smoothing import get_gaussian_kernel


class IdentityBlock(nn.Module):

    def __init__(self, in_channel, kl_size, filters):
        super(IdentityBlock, self).__init__()

        filter1, filter2, filter3 = filters
        self.cov1 = nn.Conv2d(in_channels=in_channel, out_channels=filter1, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(num_features=filter1)
        self.relu = nn.ReLU(inplace=True)

        self.cov2 = nn.Conv2d(in_channels=filter1, out_channels=filter2, kernel_size=kl_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=filter2)

        self.cov3 = nn.Conv2d(in_channels=filter2, out_channels=filter3, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(num_features=filter3)

    def forward(self, x):
        identity = self.cov1(x)
        identity = self.bn1(identity)
        identity = self.relu(identity)

        identity = self.cov2(identity)
        identity = self.bn2(identity)
        identity = self.relu(identity)

        identity = self.cov3(identity)
        identity = self.bn3(identity)

        x = x + identity
        x = self.relu(x)

        return x


class ConvBlock(nn.Module):

    def __init__(self, in_channel, kl_size, filters, stride_size=2):
        super(ConvBlock, self).__init__()

        filter1, filter2, filter3 = filters
        self.cov1 = nn.Conv2d(in_channels=in_channel, out_channels=filter1, kernel_size=1, stride=stride_size,
                              padding=0)
        self.bn1 = nn.BatchNorm2d(num_features=filter1)
        self.relu = nn.ReLU(inplace=True)

        self.cov2 = nn.Conv2d(in_channels=filter1, out_channels=filter2, kernel_size=kl_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=filter2)

        self.cov3 = nn.Conv2d(in_channels=filter2, out_channels=filter3, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(num_features=filter3)

        self.short_cut = nn.Conv2d(in_channels=in_channel, out_channels=filter3, kernel_size=1, stride=stride_size,
                                   padding=0)

    def forward(self, x):
        identity = self.cov1(x)
        identity = self.bn1(identity)
        identity = self.relu(identity)

        identity = self.cov2(identity)
        identity = self.bn2(identity)
        identity = self.relu(identity)

        identity = self.cov3(identity)
        identity = self.bn3(identity)

        short_cut = self.short_cut(x)
        x = identity + short_cut
        x = self.relu(x)

        return x


class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        self.in_channels = 3
        self.layers = [2, 3, 5, 2]
        self.cov0 = nn.Conv2d(self.in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn0 = nn.BatchNorm2d(num_features=64)
        self.relu0 = nn.ReLU(inplace=False)
        self.maxpool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.basic_layer = nn.Sequential(
            self.cov0,
            self.bn0,
            self.relu0,
            self.maxpool0
        )

        self.layer1 = nn.Sequential(
            ConvBlock(64, 3, [64, 64, 256], 1),
            IdentityBlock(256, 3, [64, 64, 256]),
            IdentityBlock(256, 3, [64, 64, 256]),
        )

        self.layer2 = nn.Sequential(
            ConvBlock(256, 3, [128, 128, 512]),
            IdentityBlock(512, 3, [128, 128, 512]),
            IdentityBlock(512, 3, [128, 128, 512]),
            IdentityBlock(512, 3, [128, 128, 512]),
        )

        self.layer3 = nn.Sequential(
            ConvBlock(512, 3, [256, 256, 1024]),
            IdentityBlock(1024, 3, [256, 256, 1024]),
            IdentityBlock(1024, 3, [256, 256, 1024]),
            IdentityBlock(1024, 3, [256, 256, 1024]),
            IdentityBlock(1024, 3, [256, 256, 1024]),
            IdentityBlock(1024, 3, [256, 256, 1024]),
        )

        self.layer4 = nn.Sequential(
            ConvBlock(1024, 3, [512, 512, 2048]),
            IdentityBlock(2048, 3, [512, 512, 2048]),
            IdentityBlock(2048, 3, [512, 512, 2048]),
        )

    def forward(self, x):
        x = self.basic_layer(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x1, x2, x3, x4]


class GenSeg(nn.Module):
    def __init__(self, eps=0.03):
        super(GenSeg, self).__init__()
        self.gen = Generator()
        self.dis = Resnet50()
        self.kernal = get_gaussian_kernel()
        self.eps = eps

    def forward(self, img):
        adv_uno = self.gen(img).detach()

        adv = self.kernal(adv_uno)
        adv = torch.min(torch.max(adv, img - self.eps), img + self.eps)
        adv = torch.clamp(adv, 0, 1)
        feat = self.dis(adv)
        return feat


if __name__ == '__main__':
    model = GenSeg()
    x = torch.rand((1, 3, 128, 128))
    out = model(x)
    for elem in out:
        print(elem.shape)

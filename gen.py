import torch
import torch.nn as nn

num_feature_maps = 64


class Generator(nn.Module):
    def __init__(self, inception=False, data_dim='high'):
        super(Generator, self).__init__()
        self.inception_flag = inception
        self.data_dim_flag = data_dim
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, num_feature_maps, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(num_feature_maps),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_feature_maps, num_feature_maps * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_feature_maps * 2),
            nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_feature_maps * 2, num_feature_maps * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_feature_maps * 4),
            nn.ReLU(True)
        )
        self.resblock1 = ResidualBlock(num_feature_maps * 4)
        self.resblock2 = ResidualBlock(num_feature_maps * 4)
        if self.data_dim_flag == 'high':
            self.resblock3 = ResidualBlock(num_feature_maps * 4)
            self.resblock4 = ResidualBlock(num_feature_maps * 4)
            self.resblock5 = ResidualBlock(num_feature_maps * 4)
            self.resblock6 = ResidualBlock(num_feature_maps * 4)

        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(num_feature_maps * 4, num_feature_maps * 2, kernel_size=3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.BatchNorm2d(num_feature_maps * 2),
            nn.ReLU(True)
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(num_feature_maps * 2, num_feature_maps, kernel_size=3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.BatchNorm2d(num_feature_maps),
            nn.ReLU(True)
        )
        self.conv_final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(num_feature_maps, 3, kernel_size=7, padding=0)
        )
        self.crop_layer = nn.ConstantPad2d((0, -1, -1, 0), 0)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        if self.data_dim_flag == 'high':
            x = self.resblock3(x)
            x = self.resblock4(x)
            x = self.resblock5(x)
            x = self.resblock6(x)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.conv_final(x)
        if self.inception_flag:
            x = self.crop_layer(x)
        return (torch.tanh(x) + 1) / 2


class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters)
        )

    def forward(self, x):
        residual = self.block(x)
        return x + residual


if __name__ == '__main__':
    Gen = Generator(data_dim='low')
    test_input = torch.rand(1, 3, 32, 32)
    print('Generator output size:', Gen(test_input).size())
    print('Generator parameters count:', sum(p.numel() for p in Gen.parameters() if p.requires_grad))


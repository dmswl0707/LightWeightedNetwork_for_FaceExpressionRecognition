import torch
from torch import nn
from torchsummary import summary


# make model architecture
# customed model

class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                                   bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ResidualBlock(nn.Module):

    def __init__(self, in_channeld, out_channels):
        super(ResidualBlock, self).__init__()

        self.residual_conv = nn.Conv2d(in_channels=in_channeld, out_channels=out_channels, kernel_size=1, stride=2,
                                       bias=False)
        self.residual_bn = nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3)

        self.sepConv1 = SeparableConv2d(in_channels=in_channeld, out_channels=out_channels, kernel_size=3, bias=False,
                                        padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3)
        self.elu = nn.ELU()

        self.sepConv2 = SeparableConv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, bias=False,
                                        padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3)
        self.maxp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        res = self.residual_conv(x)
        res = self.residual_bn(res)
        x = self.sepConv1(x)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.sepConv2(x)
        x = self.bn2(x)
        x = self.maxp(x)
        return res + x



class SELayer(nn.Module):
    def __init__(self, in_channeld, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channeld, in_channeld // reduction, bias=False),
            nn.ReLU6(inplace=True),
            nn.Linear(in_channeld // reduction, in_channeld, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channeld, out_channels, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.Conv1 = nn.Conv2d(in_channeld, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(in_channeld)
        self.relu = nn.ReLU6(inplace=True)

        self.Conv2 = nn.Conv2d(in_channeld, out_channels, 1)
        self.bn2 = nn.BatchNorm2d(in_channeld)
        self.se = SELayer(in_channeld, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.Conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.Conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Model(nn.Module):

    def __init__(self, num_classes):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, dilation=2, bias=False)
        self.bn1 = nn.BatchNorm2d(8, affine=True, momentum=0.99, eps=1e-3)
        self.relu1 = nn.ELU()
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, dilation=2, bias=False)
        self.bn2 = nn.BatchNorm2d(16, momentum=0.99, eps=1e-3)
        self.relu2 = nn.ELU()

        self.module1 = ResidualBlock(in_channeld=16, out_channels=32)
        self.module2 = SEBasicBlock(32, 32, stride=1, downsample=None, groups=1,
                                    base_width=64, dilation=1, norm_layer=None, reduction=16)
        self.module3 = ResidualBlock(in_channeld=32, out_channels=64)
        self.module4 = SEBasicBlock(64, 64, stride=1, downsample=None, groups=1,
                                    base_width=64, dilation=1, norm_layer=None, reduction=16)
        self.module4 = ResidualBlock(in_channeld=64, out_channels=128)
        self.module5 = SEBasicBlock(128, 128, stride=1, downsample=None, groups=1,
                                    base_width=64, dilation=1, norm_layer=None, reduction=16)
        self.module6 = ResidualBlock(in_channeld=128, out_channels=64)

        self.last_conv = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=3, padding=1, stride=2)
        self.avgp = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, input):
        x = input
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)
        x = self.module4(x)
        x = self.module5(x)
        x = self.module6(x)
        x = self.last_conv(x)
        x = self.avgp(x)
        x = x.view((x.shape[0], -1))
        return x

Model = Model(num_classes=10)

#print(summary(model, (3, 32, 32), device = 'cpu'))

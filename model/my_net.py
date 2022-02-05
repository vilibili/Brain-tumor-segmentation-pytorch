import torch.nn as nn
import torch
import torchvision.models as models

class ConvBnRelu(nn.Module):
    def __init__(self, in_: int, out: int, kernel:int, pad:int=1):
        super().__init__()
        self.conv = nn.Conv2d(in_, out, kernel, padding=pad)
        self.batchN = nn.BatchNorm2d(out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchN(x)
        x = self.activation(x)
        return x

class Mini_unet(nn.Module):
    def __init__(self, num_classes=4, pretrained=True, **_):
        super(Mini_unet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv_1 = ConvBnRelu(1,16,3)

        self.conv_2 = ConvBnRelu(16, 32, 3)

        self.conv_3 = ConvBnRelu(32, 64, 3)

        self.conv_4 = ConvBnRelu(64, 128, 3)

        self.conv_5 = ConvBnRelu(128, 256, 3)

        self.conv_6 = ConvBnRelu(384, 128, 3)

        self.conv_7 = ConvBnRelu(192, 64, 3)

        self.conv_8 = ConvBnRelu(96, 32, 3)

        self.conv_9 = ConvBnRelu(48, 16, 3)

        self.final = nn.Sequential(nn.Conv2d(16, num_classes, (1, 1)),nn.Softmax())

    def forward(self, x):
        x = x.float()
        x = self.conv_1(x)
        x1 = x
        x = self.pool(x)

        x = self.conv_2(x)
        x2 = x
        x = self.pool(x)

        x = self.conv_3(x)
        x3 = x
        x = self.pool(x)

        x = self.conv_4(x)
        x4 = x
        x = self.pool(x)

        x = self.conv_5(x)

        x = self.upsample(x)
        x = torch.cat([x,x4],1)
        x = self.conv_6(x)

        x = self.upsample(x)
        x = torch.cat([x, x3], 1)
        x = self.conv_7(x)

        x = self.upsample(x)
        x = torch.cat([x, x2], 1)
        x = self.conv_8(x)

        x = self.upsample(x)
        x = torch.cat([x, x1], 1)
        x = self.conv_9(x)

        x = self.final(x)

        output = x
        return output

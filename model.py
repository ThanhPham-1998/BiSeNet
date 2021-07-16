import torch
import torch.nn as nn 
from torchsummary import summary
import torchvision
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(num_features=out_channels)
<<<<<<< HEAD
        self.relu = nn.ReLU6(inplace=True)
=======
        self.relu = nn.ReLU(inplace=True)
>>>>>>> 161e9aa80ca89581d6c4db97f5707a5c9fa81069
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x 


class SP(nn.Module):
    def __init__(self):
        super().__init__()
        self.sp = nn.Sequential(*[
            ConvBlock(in_channels=3, out_channels=64),
            ConvBlock(in_channels=64, out_channels=128),
            ConvBlock(in_channels=128, out_channels=256)
        ])
    def forward(self, x):
        return self.sp(x)


class ARM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        y = self.global_pool(x)
        y = self.conv(x)
        y = self.norm(x)
        y = self.activation(x)
        return torch.mul(x, y)

class CP(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = torchvision.models.resnet34(pretrained=True)
        self.conv1 = self.model.conv1
        self.norm1 = self.model.bn1
        self.relu = self.model.relu
        self.max_pool = self.model.maxpool
        self.layer1 = self.model.layer1
        self.layer2 = self.model.layer2
        self.layer3 = self.model.layer3
        self.layer4 = self.model.layer4
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.arm_1 = ARM(in_channels=256, out_channels=256)
        self.arm_2 = ARM(in_channels=512, out_channels=512)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        pool = self.avg_pool(layer4)
        arm_1 = self.arm_1(layer3)
        arm_2 = self.arm_2(layer4)
        arm_2 = torch.mul(arm_2, pool)
        arm_2 = self.up(arm_2)
        return torch.cat((arm_1, arm_2), dim=1)


class FFM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv_block = ConvBlock(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 1))
<<<<<<< HEAD
        self.relu = nn.ReLU6(inplace=True)
=======
        self.relu = nn.ReLU(inplace=True)
>>>>>>> 161e9aa80ca89581d6c4db97f5707a5c9fa81069
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 1))
        self.activation = nn.Sigmoid()
    
    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)
        conv_block = self.conv_block(x)
        x = self.global_pool(conv_block)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.activation(x)
        mul = torch.mul(conv_block, x)
        add = mul + conv_block
        return add


class BiSiNet(nn.Module):
    def __init__(self, num_classes=2, training=False):
        super().__init__()
        self.training = training
        self.sp = SP()
        self.cp = CP()
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.ffm = FFM(in_channels=1024, out_channels=num_classes)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=8)
        self.conv = ConvBlock(in_channels=num_classes, out_channels=num_classes,
                            kernel_size=1, stride=1, padding=0)
        self.conv_sp1 = ConvBlock(in_channels=256, out_channels=num_classes, kernel_size=1, padding=0)
        self.conv_sp2 = ConvBlock(in_channels=768, out_channels=num_classes, kernel_size=1, padding=0)
        
    def forward(self, x):
        sp = self.sp(x)
        cp = self.cp(x)
        cp = self.up1(cp)
        ffm = self.ffm(cp, sp)
        if self.training:
            return self.conv(self.up2(ffm)), self.conv_sp1(self.up2(sp)), self.conv_sp2(self.up2(cp))
        return self.conv(self.up2(ffm))



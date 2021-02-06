import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
from collections import OrderedDict

def get_padding(input, output, kernel_size, stride):
    padding = ((output - 1) * stride + kernel_size - input) // 2
    return padding

class ResidualBlock(nn.Module):
    def __init__(self, size=256):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(in_channels=size, out_channels=size,
                             kernel_size=3, stride=1, padding=1, bias=True)),
            ('relu1', nn.PReLU()),
            ('c2', nn.Conv2d(in_channels=size, out_channels=size,
                             kernel_size=3, stride=1, padding=1, bias=True)),
        ]))
        self.shortcut = nn.Sequential()
        self.activate = nn.PReLU()

    def forward(self, input):
        output = self.left(input)
        output += self.shortcut(input)
        output = self.activate(output)
        return output

class JointPixel(nn.Module):
    def __init__(self, resnet_level=2,debug=False):
        super(JointPixel, self).__init__()
        self.debug = debug
        self.stage1 = nn.Sequential(OrderedDict([
            ('stage1_conv3x3 ', nn.Conv2d(in_channels=3, out_channels=64,
                                            kernel_size=3, stride=1, padding=1, bias=True)),
            ('stage1_PReLU', nn.PReLU())
            ]))
        stage2 = [ResidualBlock(size=64) for i in range(resnet_level)]
        self.stage2 = nn.Sequential(*stage2)
        self.stage3_left = nn.Sequential(OrderedDict([
            ('stage3_left_conv3x3 ', nn.Conv2d(in_channels=64, out_channels=61,
                                            kernel_size=3, stride=1, padding=1, bias=True))
            # ('stage3_left_PReLU', nn.PReLU())
            ]))
        self.stage3_right = nn.Sequential(OrderedDict([
            ('stage3_right_conv3x3 ', nn.Conv2d(in_channels=64, out_channels=3,
                                            kernel_size=3, stride=1, padding=1, bias=True))
            # ('stage3_right_PReLU', nn.PReLU())
            ]))
        stage4 = [ResidualBlock(size=64) for i in range(resnet_level)]
        self.stage4 = nn.Sequential(*stage4)
        self.stage5 = nn.Sequential(OrderedDict([
            ('stage5_conv3x3 ', nn.Conv2d(in_channels=64, out_channels=3,
                                            kernel_size=3, stride=1, padding=1, bias=True))
            ]))
        self.shortcut = nn.Sequential()

    def forward(self, input):
        stage1_output = self.stage1(input)
        stage2_output = self.stage2(stage1_output)
        stage3_left_output = self.stage3_left(stage2_output)
        stage3_right_intput = self.stage3_right(stage2_output)
        stage3_right_output = stage3_right_intput + self.shortcut(input)
        stage3_output = torch.cat((stage3_left_output,stage3_right_output),dim=1)
        stage4_output = self.stage4(stage3_output)
        stage5_input  = self.stage5(stage4_output)
        stage5_output = stage5_input + self.shortcut(stage3_right_output)
        if self.debug:
            return stage3_right_output,stage5_output,stage5_input
        else:
            return stage5_output
        
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

class JointPixel_fusion(nn.Module):
    def __init__(self, resnet_level=2,debug=False,fusion_degree=2):
        super(JointPixel_fusion, self).__init__()
        self.fusion_degree = fusion_degree
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
            ]))
        self.w1 = nn.ParameterList([nn.Parameter(torch.randn(1)) for _ in range(fusion_degree)]).cuda()
        self.stage3_right = nn.Sequential(OrderedDict([
            ('stage3_right_conv3x3 ', nn.Conv2d(in_channels=64, out_channels=3,
                                            kernel_size=3, stride=1, padding=1, bias=True))
            ]))
        self.w2 = nn.ParameterList([nn.Parameter(torch.randn(1)) for _ in range(fusion_degree)]).cuda()
        stage4 = [ResidualBlock(size=64) for i in range(resnet_level)]
        self.stage4 = nn.Sequential(*stage4)
        self.stage5 = nn.Sequential(OrderedDict([
            ('stage5_conv3x3 ', nn.Conv2d(in_channels=64, out_channels=3,
                                            kernel_size=3, stride=1, padding=1, bias=True))
            ]))
        self.shortcut = nn.Sequential()

    def forward(self, input):
        input_list = [input[:,i:i+3] for i in range(self.fusion_degree)]

        fusion_stage3_left_output = []
        fusion_stage3_right_output = []
        for i,input in enumerate(input_list):
            stage1_output = self.stage1(input)
            stage2_output = self.stage2(stage1_output)
            stage3_left_output = self.stage3_left(stage2_output)
            stage3_right_intput = self.stage3_right(stage2_output)
            stage3_right_output = stage3_right_intput + self.shortcut(input)
            fusion_stage3_left_output.append(stage3_left_output*self.w1[i].expand_as(stage3_left_output))
            fusion_stage3_right_output.append(stage3_right_output*self.w2[i].expand_as(stage3_right_output))
        fusion_stage3_left_output = torch.stack(fusion_stage3_left_output, dim=0).sum(dim=0)
        fusion_stage3_right_output = torch.stack(fusion_stage3_right_output, dim=0).sum(dim=0)

        stage3_output = torch.cat((fusion_stage3_left_output,fusion_stage3_right_output),dim=1)
        stage4_output = self.stage4(stage3_output)
        stage5_input  = self.stage5(stage4_output)
        stage5_output = stage5_input + self.shortcut(stage3_right_output)
        if self.debug:
            return stage3_right_output,stage5_output,stage5_input
        else:
            return stage5_output


if __name__ == '__main__':
    net = JointPixel_fusion()
    state_dict = net.state_dict()
    torch.save(state_dict,'last_ckp.pth')
    state_dict = torch.load('last_ckp.pth')
    print(state_dict.keys())
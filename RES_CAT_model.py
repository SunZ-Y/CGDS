import torch
from torch import nn as nn
import torch.nn.functional as F


# conv1 7 x 7 64 stride=2
def Conv1(channel_in, channel_out, stride=2):
    return nn.Sequential(
        nn.Conv2d(
            channel_in,
            channel_out,
            kernel_size=7,
            stride=stride,
            padding=3,
            bias=False
        ),
        nn.BatchNorm2d(channel_out),
        # 会改变输入数据的值
        # 节省反复申请与释放内存的空间与时间
        # 只是将原来的地址传递，效率更好
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
    )


# 构建ResNet18-34的网络基础模块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        self.shortcut(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()

        self.in_planes = 64
        self.in_planes1 = 64
        self.conv1 = nn.Conv2d(5, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(5, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        # 逐层搭建ResNet
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(10752, 120)

        self.layer5 = self._make_layer1(block, 64, num_blocks[0], stride=1)
        self.layer6 = self._make_layer1(block, 128, num_blocks[1], stride=2)
        self.layer7 = self._make_layer1(block, 256, num_blocks[2], stride=2)
        self.layer8 = self._make_layer1(block, 512, num_blocks[3], stride=2)
        self.linear2 = nn.Linear(53760, 120)

        self.linear1 = nn.Linear(120, 30)
        self.drop = nn.Dropout(p=0.5)
        self.linear_1 = nn.Linear(154, 60)
        self.linear_2 = nn.Linear(60, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _make_layer1(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes1, planes, stride))
            self.in_planes1 = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, y, z):


        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)



        z = F.relu(self.bn2(self.conv2(z)))
        z = self.layer5(z)
        z = self.layer6(z)
        z = self.layer7(z)
        z = self.layer8(z)
        z = F.avg_pool2d(z, 4)
        z = z.view(z.size(0), -1)
        z = self.linear2(z)
        z = self.linear1(z)

        out = torch.cat((out, y, z), dim=1)
        out = self.drop(out)
        out = self.linear_1(out)
        out = self.drop(out)
        out = self.linear_2(out)
        return out


def Res_Cat():
    return ResNet(BasicBlock, [2, 2, 2, 2])



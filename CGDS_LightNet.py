import torch.nn.functional as F
from torchvision.ops import DeformConv2d
from ECA import *

eca = ECA_layer(channel=5)
eca1 = ECA_layer(channel=5)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dilation=2):
        super(BasicBlock, self).__init__()

        # 改为空洞卷积
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # 改为空洞卷积
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation, dilation=dilation, bias=False)
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
        out += self.shortcut(x)  # Shortcut should be added to the output
        out = F.relu(out)
        return out


class BasicBlock1(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dilation1=2, kernel_size1=3, kernel_size2=5):
        super(BasicBlock1, self).__init__()
        # 改为空洞卷积
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=kernel_size1, stride=stride, padding=dilation1, dilation=dilation1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # 第一次空洞卷积，卷积核大小为 kernel_size1，空洞率为 dilation1
        self.conv3 = nn.Conv2d(
            in_planes, planes, kernel_size=kernel_size2, stride=stride, padding=2*dilation1, dilation=dilation1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        # 改为空洞卷积
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation1, dilation=dilation1, bias=False)
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
        out_1 = F.relu(self.bn3(self.conv3(x)))
        out = out + out_1
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Shortcut should be added to the output
        out = F.relu(out)
        return out



class CGDS_Light(nn.Module):
    def __init__(self, block,block1, num_blocks, num_classes=10):
        super(CGDS_Light, self).__init__()

        self.eca = eca
        self.eca1 = eca1
        self.in_planes = 12
        self.in_planes1 = 12
        self.conv1 = nn.Conv2d(5, 12, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(12)

        self.conv2 = nn.Conv2d(5, 12, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(12)


        self.layer1 = self._make_layer(block, 12, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 16, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 20, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 24, num_blocks[3], stride=2)
        self.linear = nn.Linear(504, 120)

        self.layer5 = self._make_layer1(block1, 12, num_blocks[0], stride=1)
        self.layer6 = self._make_layer1(block1, 16, num_blocks[1], stride=2)
        self.layer7 = self._make_layer1(block1, 20, num_blocks[2], stride=2)
        self.layer8 = self._make_layer1(block1, 24, num_blocks[3], stride=2)
        self.linear2 = nn.Linear(2520, 120)

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
        x = self.eca(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        z = self.eca1(z)
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


def CGDS_LightNet():
    return CGDS_Light(BasicBlock, BasicBlock1,[2, 2, 2, 2])




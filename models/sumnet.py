'''
SumNet in PyTorch
Author: Cai Shaofeng
Data:   2017.10.2
'''

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat([x, out], 1)
        return out

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, growth_rate, layer_idx):
        super(BottleneckBlock, self).__init__()
        input_planes = in_planes+(growth_rate if layer_idx>0 else 0)
        self.bn1 = nn.BatchNorm2d(input_planes)
        self.conv1 = nn.Conv2d(input_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

        self.in_planes = in_planes
        self.growth_rate = growth_rate
        self.layer_idx = layer_idx

    def forward(self, x):
        input = x
        if self.layer_idx > 1:
            input = torch.cat([x[:, :self.in_planes, :, :], x[:, -self.growth_rate:, :, :]], dim=1)
        out = self.conv1(F.relu(self.bn1(input)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([x, out], 1)
        return out

class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out

class SumBlock(nn.Module):
    def __init__(self, n_layers, in_planes, growth_rate, block):
        super(SumBlock, self).__init__()
        self.layer = self._make_layer(n_layers, in_planes, growth_rate, block)

    def _make_layer(self, n_layers, in_planes, growth_rate, block):
        layers = []
        for layer_idx in range(n_layers):
            layers.append(block(in_planes, growth_rate, layer_idx))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)



class SumNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(SumNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.sum_layer, num_planes = self._make_sum_layers(num_planes, block, nblocks, growth_rate, reduction)

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_sum_layers(self, in_planes, block, nblocks, growth_rate, reduction):
        layers = []
        for num_blocks in nblocks:
            layers.append(SumBlock(num_blocks, in_planes, growth_rate, block))
            in_planes += num_blocks*growth_rate
            out_planes = int(math.floor(in_planes*reduction))
            layers.append(Transition(in_planes, out_planes))
            in_planes = out_planes
        return nn.Sequential(*layers), in_planes

    def forward(self, x):
        out = self.conv1(x)
        out = self.sum_layer(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), out.size(-1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def test_sumnet():
    net = SumNet(BottleneckBlock, [3,  ], growth_rate=3)
    x = torch.randn(1,3,32,32)
    y = net(Variable(x))
    print(y)

# test_sumnet()

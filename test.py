from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from torch.autograd import Variable

from data_loader import *


a = torch.FloatTensor(2, 3, 4).fill_(1)
b = torch.FloatTensor(2, 5,4 )
print(a, b)

print(torch.cat([b[:, :2, :], b[:, -1:, :]], dim=1))
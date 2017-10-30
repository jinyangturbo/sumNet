import torch
import torch.nn as nn
from torch.autograd import Variable

class DropoutLayer(nn.Module):
    def __init__(self, in_planes, p):
        super(DropoutLayer, self).__init__()
        assert p < 1.
        self.p = p
        self.in_planes = in_planes
        self.prob_tensor = torch.FloatTensor(1).fill_(1-self.p).expand((self.in_planes))
        # print(self.p)

    def forward(self, x):
        if self.training==False: return x
        # batch shared dropout mask
        self.mask = torch.bernoulli(self.prob_tensor)
        view_size = [1, self.in_planes] + [1] * (len(x.size()) - 2)
        self.input_mask = Variable((self.mask / (1. - self.p)).view(view_size).expand_as(x)).cuda()
        return x*self.input_mask



def test():
    input = Variable(torch.FloatTensor(3, 4, 5, 6).random_(10))
    print(input)

    dropout_layer = DropoutLayer(4, 0.4)
    print(dropout_layer(input))


# test()
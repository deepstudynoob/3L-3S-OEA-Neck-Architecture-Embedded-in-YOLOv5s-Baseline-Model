import torch
import torchvision.ops
from torch import nn
import math


class DCNv2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1):
        super(DCNv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding

        # init weight and bias
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        # offset conv
        self.conv_offset_mask = nn.Conv2d(in_channels,
                                          3 * kernel_size * kernel_size,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=self.padding,
                                          bias=True)

        # init
        self.reset_parameters()
        self._init_weight()

    def reset_parameters(self):
        n = self.in_channels * (self.kernel_size ** 2)
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def _init_weight(self):
        # init offset_mask conv
        nn.init.constant_(self.conv_offset_mask.weight, 0.)
        nn.init.constant_(self.conv_offset_mask.bias, 0.)

    def forward(self, x):
        out = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.weight,
                                          bias=self.bias,
                                          padding=self.padding,
                                          mask=mask,
                                          stride=self.stride)
        return x


model = nn.Sequential(
    DCNv2(3, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2, 2),
    DCNv2(32, 32, kernel_size=3, stride=1, padding=1),
    DCNv2(32, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2, 2),
    DCNv2(64, 64, kernel_size=3, stride=1, padding=1),
    DCNv2(64, 128, kernel_size=3, stride=1, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2, 2),
    DCNv2(128, 128, kernel_size=3, stride=1, padding=1),
    DCNv2(128, 256, kernel_size=3, stride=1, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2, 2)
)
x = torch.randn(2, 3, 64, 64)
y = model(x)
print(x.size())
print(y.size())
"""
torch.Size([2, 3, 64, 64])
torch.Size([2, 256, 4, 4])
"""

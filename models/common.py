# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Common modules
"""

import ast
import contextlib
import json
import math
import platform
import warnings
import zipfile
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path
from urllib.parse import urlparse
import torch.nn.functional as F
from torchvision.ops import deform_conv2d
import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from IPython.display import display
from PIL import Image
from torch.cuda import amp

from utils import TryExcept
from utils.dataloaders import exif_transpose, letterbox
from utils.general import (LOGGER, ROOT, Profile, check_requirements, check_suffix, check_version, colorstr,
                           increment_path, is_notebook, make_divisible, non_max_suppression, scale_boxes, xywh2xyxy,
                           xyxy2xywh, yaml_load)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import copy_attr, smart_inference_mode

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

 
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module): #标准的卷积 conv+BN+hardswish
    """在Focus、Bottleneck、BottleneckCSP、C3、SPP、DWConv、TransformerBloc等模块中调用
    Standard convolution  conv+BN+act
    :params c1: 输入的channel值
    :params c2: 输出的channel值
    :params k: 卷积的kernel_size
    :params s: 卷积的stride
    :params p: 卷积的padding  一般是None  可以通过autopad自行计算需要pad的padding数
    :params g: 卷积的groups数  =1就是普通的卷积  >1就是深度可分离卷积
    :params act: 激活函数类型   True就是SiLU()/Swish   False就是不使用激活函数
                 类型是nn.Module就使用传进来的激活函数类型
    """
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):  # ch_in, ch_out, 卷积核kernel, 步长stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x): #正向传播
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """用于Model类的fuse函数
        融合conv+bn 加速推理 一般用于测试/验证阶段
        """
        return self.act(self.conv(x))


class DWConv(Conv): #深度可分离卷积网络
    # Depth-wise convolution
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    # Depth-wise transpose convolution
    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    """
     Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
     视频: https://www.bilibili.com/video/BV1Di4y1c7Zm?p=5&spm_id_from=pageDriver
          https://www.bilibili.com/video/BV1v3411r78R?from=search&seid=12070149695619006113
     这部分相当于原论文中的单个Encoder部分(只移除了两个Norm部分, 其他结构和原文中的Encoding一模一样)
    """
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        # 输入: query、key、value
        # 输出: 0 attn_output 即通过self-attention之后，从每一个词语位置输出来的attention 和输入的query它们形状一样的
        #      1 attn_output_weights 即attention weights 每一个单词和任意另一个单词之间都会产生一个weight
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        # 多头注意力机制 + 残差(这里移除了LayerNorm for better performance)
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        # feed forward 前馈神经网络 + 残差(这里移除了LayerNorm for better performance)
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    """
    Vision Transformer https://arxiv.org/abs/2010.11929
    视频: https://www.bilibili.com/video/BV1Di4y1c7Zm?p=5&spm_id_from=pageDriver
         https://www.bilibili.com/video/BV1v3411r78R?from=search&seid=12070149695619006113
    这部分相当于原论文中的Encoders部分 只替换了一些编码方式和最后Encoders出来数据处理方式
    """
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding 位置编码
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2 #输出channel

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class Bottleneck(nn.Module):
    # Standard bottleneck
    """在BottleneckCSP和yolo.py的parse_model中调用
    Standard bottleneck  Conv+Conv+shortcut
    :params c1: 第一个卷积的输入channel
    :params c2: 第二个卷积的输出channel
    :params shortcut: bool 是否有shortcut连接 默认是True
    :params g: 卷积分组的个数  =1就是普通卷积  >1就是深度可分离卷积
    :params e: expansion ratio  e*c2就是第一个卷积的输出channel=第二个卷积的输入channel
    """
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    """在C3模块和yolo.py的parse_model模块调用
            CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
            :params c1: 整个BottleneckCSP的输入channel
            :params c2: 整个BottleneckCSP的输出channel
            :params n: 有n个Bottleneck
            :params shortcut: bool Bottleneck中是否有shortcut，默认True
            :params g: Bottleneck中的3x3卷积类型  =1普通卷积  >1深度可分离卷积
            :params e: expansion ratio c2xe=中间其他所有层的卷积核个数/中间所有层的输入输出channel数
            """
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n))) #*把list拆分为一个个元素
        # 叠加n次Bottleneck

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    """在C3TR模块和yolo.py的parse_model模块调用
    CSP Bottleneck with 3 convolutions
    :params c1: 整个BottleneckCSP的输入channel
    :params c2: 整个BottleneckCSP的输出channel
    :params n: 有n个Bottleneck
    :params shortcut: bool Bottleneck中是否有shortcut，默认True
    :params g: Bottleneck中的3x3卷积类型  =1普通卷积  >1深度可分离卷积
    :params e: expansion ratio c2xe=中间其他所有层的卷积核个数/中间所有层的输入输出channel数
    """
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        # 实验性 CrossConv
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    # C3 module with cross-convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)))


class C3TR(C3):
    # C3 module with TransformerBlock()
    """
    这部分是根据上面的C3结构改编而来的, 将原先的Bottleneck替换为调用TransformerBlock模块
    """
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class SPP(nn.Module): #金字塔池化
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    """在yolo.py的parse_model模块调用
    空间金字塔池化 Spatial pyramid pooling layer used in YOLOv3-SPP
    :params c1: SPP模块的输入channel
    :params c2: SPP模块的输出channel
    :params k: 保存着三个maxpool的卷积核大小 默认是(5, 9, 13)
    """
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            # print(y2.shape)
            # print(y1.shape)
            # print(x.shape)
            z1 = self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))
            # print(z1.shape)
            return z1



class DeepWise_Pool(torch.nn.MaxPool1d):
    def __init__(self, channels, isize):
        super(DeepWise_Pool, self).__init__(channels)
        self.kernel_size = channels
        self.stride = isize

    def forward(self, input):
        n, c, w, h = input.size()
        input = input.view(n,c,w*h).permute(0,2,1)
        pooled =  torch.nn.functional.max_pool1d(input, self.kernel_size, self.stride, self.padding,
                                                 self.dilation, self.ceil_mode,self.return_indices)
        _, _, c = pooled.size()
        pooled = pooled.permute(0,2,1)
        return pooled.view(n,c,w,h).view(w, h)






class SPPF_1(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 6, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)




    def forward(self, x):
        shape = [None] * 2
        # print(x.shape)
        x = self.cv1(x)
        # print(x.shape)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            # print(y1.shape)
            y2 = self.m(y1)
            # print(y2.shape)
            y3 = self.m(y2)
            shape = y3.shape
            # print(shape[0],shape[1],shape[2],shape[3])
            # print(y3.shape)

        # Global Average Pooling
        gap = F.adaptive_avg_pool2d(x, (shape[2], shape[3]))
        # print(gap.shape)
        # gap = gap.view(gap.size(0), -1)  # Flatten to (batch_size, c_)
        # print(gap.shape)
        # Global Max Pooling
        gmp = F.adaptive_max_pool2d(x, (shape[2], shape[3]))
        # print(gmp.shape)
        # gmp = gmp.view(gmp.size(0), -1)  # Flatten to (batch_size, c_)

        # Concatenate original x, y1, y2, y3, gap, and gmp
        y = torch.cat((x, y1, y2, y3, gap, gmp), 1)
        # print(x.shape)
        # print(y1.shape)
        # print(y2.shape)
        # print(y3.shape)
        # print(gap.shape)
        # print(gmp.shape)
        # print(y.shape)

        # # Concatenate GAP result
        # gap = gap.unsqueeze(-1).unsqueeze(-1)  # Reshape to (batch_size, c_, 1, 1)
        #
        # gap = gap.expand(-1, -1, y.size(2), y.size(3))  # Expand to match spatial dimensions
        # y = torch.cat((y, gap), 1)
        #
        # # Concatenate GMP result
        # gmp = gmp.unsqueeze(-1).unsqueeze(-1)  # Reshape to (batch_size, c_, 1, 1)
        # gmp = gmp.expand(-1, -1, y.size(2), y.size(3))  # Expand to match spatial dimensions
        # y = torch.cat((y, gmp), 1)
        y = self.cv2(y)
        # print(y.shape)
        return y


class Focus(nn.Module): #把宽和高填充整合到c空间中
    # Focus wh information into c-space
    """在yolo.py的parse_model函数中被调用
    理论：从高分辨率图像中，周期性的抽出像素点重构到低分辨率图像中，即将图像相邻的四个位置进行堆叠，
        聚焦wh维度信息到c通道空，提高每个点感受野，并减少原始信息的丢失，该模块的设计主要是减少计算量加快速度。
    Focus wh information into c-space 把宽度w和高度h的信息整合到c空间中
    先做4个slice 再concat 最后再做Conv
    slice后 (b,c1,w,h) -> 分成4个slice 每个slice(b,c1,w/2,h/2)
    concat(dim=1)后 4个slice(b,c1,w/2,h/2)) -> (b,4c1,w/2,h/2)
    conv后 (b,4c1,w/2,h/2) -> (b,c2,w/2,h/2)
    :params c1: slice后的channel
    :params c2: Focus最终输出的channel
    :params k: 最后卷积的kernel
    :params s: 最后卷积的stride
    :params p: 最后卷积的padding
    :params g: 最后卷积的分组情况  =1普通卷积  >1深度可分离卷积
    :params act: bool激活函数类型  默认True:SiLU()/Swish  False:不用激活函数
    """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    """用在yolo.py的parse_model模块 用的不多
    改变输入特征的shape 将w和h维度(缩小)的数据收缩到channel维度上(放大)
    Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    """
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain' # 1 64 80 80
        s = self.gain # 2
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        # permute: 改变tensor的维度顺序
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    """用在yolo.py的parse_model模块  用的不多
    改变输入特征的shape 将channel维度(变小)的数据扩展到W和H维度(变大)
    Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    """
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s ** 2, h * s, w * s)  # x(1,16,160,160)


class Concat(nn.Module): #这个函数是讲自身（a list of tensors）按照某个维度进行concat，常用来合并前后两个feature map，也就是上面yolov5s结构图中的Concat。
    # Concatenate a list of tensors along dimension
    """在yolo.py的parse_model模块调用
    Concatenate a list of tensors along dimension
    :params dimension: 沿着哪个维度进行concat
    """
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        # print(x[0].shape)
        # print(x[1].shape)
        # print(x[2].shape)
        out = torch.cat(x, self.d)
        # print(out.shape)
        return out



class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights='yolov5s.pt', device=torch.device('cpu'), dnn=False, data=None, fp16=False, fuse=True):
        # Usage:
        #   PyTorch:              weights = *.pt
        #   TorchScript:                    *.torchscript
        #   ONNX Runtime:                   *.onnx
        #   ONNX OpenCV DNN:                *.onnx --dnn
        #   OpenVINO:                       *_openvino_model
        #   CoreML:                         *.mlmodel
        #   TensorRT:                       *.engine
        #   TensorFlow SavedModel:          *_saved_model
        #   TensorFlow GraphDef:            *.pb
        #   TensorFlow Lite:                *.tflite
        #   TensorFlow Edge TPU:            *_edgetpu.tflite
        #   PaddlePaddle:                   *_paddle_model
        from models.experimental import attempt_download, attempt_load  # scoped to avoid circular import

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, triton = self._model_type(w)
        fp16 &= pt or jit or onnx or engine  # FP16
        nhwc = coreml or saved_model or pb or tflite or edgetpu  # BHWC formats (vs torch BCWH)
        stride = 32  # default stride
        cuda = torch.cuda.is_available() and device.type != 'cpu'  # use CUDA
        if not (pt or triton):
            w = attempt_download(w)  # download if not local

        if pt:  # PyTorch
            model = attempt_load(weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse)
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        elif jit:  # TorchScript
            LOGGER.info(f'Loading {w} for TorchScript inference...')
            extra_files = {'config.txt': ''}  # model metadata
            model = torch.jit.load(w, _extra_files=extra_files, map_location=device)
            model.half() if fp16 else model.float()
            if extra_files['config.txt']:  # load metadata dict
                d = json.loads(extra_files['config.txt'],
                               object_hook=lambda d: {int(k) if k.isdigit() else k: v
                                                      for k, v in d.items()})
                stride, names = int(d['stride']), d['names']
        elif dnn:  # ONNX OpenCV DNN
            LOGGER.info(f'Loading {w} for ONNX OpenCV DNN inference...')
            check_requirements('opencv-python>=4.5.4')
            net = cv2.dnn.readNetFromONNX(w)
        elif onnx:  # ONNX Runtime
            LOGGER.info(f'Loading {w} for ONNX Runtime inference...')
            check_requirements(('onnx', 'onnxruntime-gpu' if cuda else 'onnxruntime'))
            import onnxruntime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            session = onnxruntime.InferenceSession(w, providers=providers)
            output_names = [x.name for x in session.get_outputs()]
            meta = session.get_modelmeta().custom_metadata_map  # metadata
            if 'stride' in meta:
                stride, names = int(meta['stride']), eval(meta['names'])
        elif xml:  # OpenVINO
            LOGGER.info(f'Loading {w} for OpenVINO inference...')
            check_requirements('openvino')  # requires openvino-dev: https://pypi.org/project/openvino-dev/
            from openvino.runtime import Core, Layout, get_batch
            ie = Core()
            if not Path(w).is_file():  # if not *.xml
                w = next(Path(w).glob('*.xml'))  # get *.xml file from *_openvino_model dir
            network = ie.read_model(model=w, weights=Path(w).with_suffix('.bin'))
            if network.get_parameters()[0].get_layout().empty:
                network.get_parameters()[0].set_layout(Layout("NCHW"))
            batch_dim = get_batch(network)
            if batch_dim.is_static:
                batch_size = batch_dim.get_length()
            executable_network = ie.compile_model(network, device_name="CPU")  # device_name="MYRIAD" for Intel NCS2
            stride, names = self._load_metadata(Path(w).with_suffix('.yaml'))  # load metadata
        elif engine:  # TensorRT
            LOGGER.info(f'Loading {w} for TensorRT inference...')
            import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
            check_version(trt.__version__, '7.0.0', hard=True)  # require tensorrt>=7.0.0
            if device.type == 'cpu':
                device = torch.device('cuda:0')
            Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
            logger = trt.Logger(trt.Logger.INFO)
            with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            context = model.create_execution_context()
            bindings = OrderedDict()
            output_names = []
            fp16 = False  # default updated below
            dynamic = False
            for i in range(model.num_bindings):
                name = model.get_binding_name(i)
                dtype = trt.nptype(model.get_binding_dtype(i))
                if model.binding_is_input(i):
                    if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                        dynamic = True
                        context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
                    if dtype == np.float16:
                        fp16 = True
                else:  # output
                    output_names.append(name)
                shape = tuple(context.get_binding_shape(i))
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            batch_size = bindings['images'].shape[0]  # if dynamic, this is instead max batch size
        elif coreml:  # CoreML
            LOGGER.info(f'Loading {w} for CoreML inference...')
            import coremltools as ct
            model = ct.models.MLModel(w)
        elif saved_model:  # TF SavedModel
            LOGGER.info(f'Loading {w} for TensorFlow SavedModel inference...')
            import tensorflow as tf
            keras = False  # assume TF1 saved_model
            model = tf.keras.models.load_model(w) if keras else tf.saved_model.load(w)
        elif pb:  # GraphDef https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            LOGGER.info(f'Loading {w} for TensorFlow GraphDef inference...')
            import tensorflow as tf

            def wrap_frozen_graph(gd, inputs, outputs):
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped
                ge = x.graph.as_graph_element
                return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure(ge, outputs))

            def gd_outputs(gd):
                name_list, input_list = [], []
                for node in gd.node:  # tensorflow.core.framework.node_def_pb2.NodeDef
                    name_list.append(node.name)
                    input_list.extend(node.input)
                return sorted(f'{x}:0' for x in list(set(name_list) - set(input_list)) if not x.startswith('NoOp'))

            gd = tf.Graph().as_graph_def()  # TF GraphDef
            with open(w, 'rb') as f:
                gd.ParseFromString(f.read())
            frozen_func = wrap_frozen_graph(gd, inputs="x:0", outputs=gd_outputs(gd))
        elif tflite or edgetpu:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
            try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
                from tflite_runtime.interpreter import Interpreter, load_delegate
            except ImportError:
                import tensorflow as tf
                Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate,
            if edgetpu:  # TF Edge TPU https://coral.ai/software/#edgetpu-runtime
                LOGGER.info(f'Loading {w} for TensorFlow Lite Edge TPU inference...')
                delegate = {
                    'Linux': 'libedgetpu.so.1',
                    'Darwin': 'libedgetpu.1.dylib',
                    'Windows': 'edgetpu.dll'}[platform.system()]
                interpreter = Interpreter(model_path=w, experimental_delegates=[load_delegate(delegate)])
            else:  # TFLite
                LOGGER.info(f'Loading {w} for TensorFlow Lite inference...')
                interpreter = Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            # load metadata
            with contextlib.suppress(zipfile.BadZipFile):
                with zipfile.ZipFile(w, "r") as model:
                    meta_file = model.namelist()[0]
                    meta = ast.literal_eval(model.read(meta_file).decode("utf-8"))
                    stride, names = int(meta['stride']), meta['names']
        elif tfjs:  # TF.js
            raise NotImplementedError('ERROR: YOLOv5 TF.js inference is not supported')
        elif paddle:  # PaddlePaddle
            LOGGER.info(f'Loading {w} for PaddlePaddle inference...')
            check_requirements('paddlepaddle-gpu' if cuda else 'paddlepaddle')
            import paddle.inference as pdi
            if not Path(w).is_file():  # if not *.pdmodel
                w = next(Path(w).rglob('*.pdmodel'))  # get *.pdmodel file from *_paddle_model dir
            weights = Path(w).with_suffix('.pdiparams')
            config = pdi.Config(str(w), str(weights))
            if cuda:
                config.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=0)
            predictor = pdi.create_predictor(config)
            input_handle = predictor.get_input_handle(predictor.get_input_names()[0])
            output_names = predictor.get_output_names()
        elif triton:  # NVIDIA Triton Inference Server
            LOGGER.info(f'Using {w} as Triton Inference Server...')
            check_requirements('tritonclient[all]')
            from utils.triton import TritonRemoteModel
            model = TritonRemoteModel(url=w)
            nhwc = model.runtime.startswith("tensorflow")
        else:
            raise NotImplementedError(f'ERROR: {w} is not a supported format')

        # class names
        if 'names' not in locals():
            names = yaml_load(data)['names'] if data else {i: f'class{i}' for i in range(999)}
        if names[0] == 'n01440764' and len(names) == 1000:  # ImageNet
            names = yaml_load(ROOT / 'data/ImageNet.yaml')['names']  # human-readable names

        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)

        if self.pt:  # PyTorch
            y = self.model(im, augment=augment, visualize=visualize) if augment or visualize else self.model(im)
        elif self.jit:  # TorchScript
            y = self.model(im)
        elif self.dnn:  # ONNX OpenCV DNN
            im = im.cpu().numpy()  # torch to numpy
            self.net.setInput(im)
            y = self.net.forward()
        elif self.onnx:  # ONNX Runtime
            im = im.cpu().numpy()  # torch to numpy
            y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
        elif self.xml:  # OpenVINO
            im = im.cpu().numpy()  # FP32
            y = list(self.executable_network([im]).values())
        elif self.engine:  # TensorRT
            if self.dynamic and im.shape != self.bindings['images'].shape:
                i = self.model.get_binding_index('images')
                self.context.set_binding_shape(i, im.shape)  # reshape if dynamic
                self.bindings['images'] = self.bindings['images']._replace(shape=im.shape)
                for name in self.output_names:
                    i = self.model.get_binding_index(name)
                    self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))
            s = self.bindings['images'].shape
            assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            self.binding_addrs['images'] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = [self.bindings[x].data for x in sorted(self.output_names)]
        elif self.coreml:  # CoreML
            im = im.cpu().numpy()
            im = Image.fromarray((im[0] * 255).astype('uint8'))
            # im = im.resize((192, 320), Image.ANTIALIAS)
            y = self.model.predict({'image': im})  # coordinates are xywh normalized
            if 'confidence' in y:
                box = xywh2xyxy(y['coordinates'] * [[w, h, w, h]])  # xyxy pixels
                conf, cls = y['confidence'].max(1), y['confidence'].argmax(1).astype(np.float)
                y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
            else:
                y = list(reversed(y.values()))  # reversed for segmentation models (pred, proto)
        elif self.paddle:  # PaddlePaddle
            im = im.cpu().numpy().astype(np.float32)
            self.input_handle.copy_from_cpu(im)
            self.predictor.run()
            y = [self.predictor.get_output_handle(x).copy_to_cpu() for x in self.output_names]
        elif self.triton:  # NVIDIA Triton Inference Server
            y = self.model(im)
        else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
            im = im.cpu().numpy()
            if self.saved_model:  # SavedModel
                y = self.model(im, training=False) if self.keras else self.model(im)
            elif self.pb:  # GraphDef
                y = self.frozen_func(x=self.tf.constant(im))
            else:  # Lite or Edge TPU
                input = self.input_details[0]
                int8 = input['dtype'] == np.uint8  # is TFLite quantized uint8 model
                if int8:
                    scale, zero_point = input['quantization']
                    im = (im / scale + zero_point).astype(np.uint8)  # de-scale
                self.interpreter.set_tensor(input['index'], im)
                self.interpreter.invoke()
                y = []
                for output in self.output_details:
                    x = self.interpreter.get_tensor(output['index'])
                    if int8:
                        scale, zero_point = output['quantization']
                        x = (x.astype(np.float32) - zero_point) * scale  # re-scale
                    y.append(x)
            y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
            y[0][..., :4] *= [w, h, w, h]  # xywh normalized to pixels

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        # Warmup model by running inference once
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton
        if any(warmup_types) and (self.device.type != 'cpu' or self.triton):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.jit else 1):  #
                self.forward(im)  # warmup

    @staticmethod
    def _model_type(p='path/to/model.pt'):
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
        # types = [pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle]
        from export import export_formats
        from utils.downloads import is_url
        sf = list(export_formats().Suffix)  # export suffixes
        if not is_url(p, check=False):
            check_suffix(p, sf)  # checks
        url = urlparse(p)  # if url may be Triton inference server
        types = [s in Path(p).name for s in sf]
        types[8] &= not types[9]  # tflite &= not edgetpu
        triton = not any(types) and all([any(s in url.scheme for s in ["http", "grpc"]), url.netloc])
        return types + [triton]

    @staticmethod
    def _load_metadata(f=Path('path/to/meta.yaml')):
        # Load metadata from meta.yaml if it exists
        if f.exists():
            d = yaml_load(f)
            return d['stride'], d['names']  # assign stride, names
        return None, None


class AutoShape(nn.Module): #自动调整大小
    # YOLOv5 input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    agnostic = False  # NMS class-agnostic
    multi_label = False  # NMS multiple labels per box
    classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    max_det = 1000  # maximum number of detections per image
    amp = False  # Automatic Mixed Precision (AMP) inference

    def __init__(self, model, verbose=True):
        super().__init__()
        if verbose:
            LOGGER.info('Adding AutoShape... ')
        copy_attr(self, model, include=('yaml', 'nc', 'hyp', 'names', 'stride', 'abc'), exclude=())  # copy attributes
        self.dmb = isinstance(model, DetectMultiBackend)  # DetectMultiBackend() instance
        self.pt = not self.dmb or model.pt  # PyTorch model
        self.model = model.eval()
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
            m.inplace = False  # Detect.inplace=False for safe multithread inference
            m.export = True  # do not output loss values

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    @smart_inference_mode()
    def forward(self, ims, size=640, augment=False, profile=False):
        # Inference from various sources. For size(height=640, width=1280), RGB images example inputs are:
        #   file:        ims = 'data/images/zidane.jpg'  # str or PosixPath
        #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        dt = (Profile(), Profile(), Profile())
        with dt[0]:
            if isinstance(size, int):  # expand
                size = (size, size)
            p = next(self.model.parameters()) if self.pt else torch.empty(1, device=self.model.device)  # param
            autocast = self.amp and (p.device.type != 'cpu')  # Automatic Mixed Precision (AMP) inference
            if isinstance(ims, torch.Tensor):  # torch
                with amp.autocast(autocast):
                    return self.model(ims.to(p.device).type_as(p), augment=augment)  # inference

            # Pre-process
            n, ims = (len(ims), list(ims)) if isinstance(ims, (list, tuple)) else (1, [ims])  # number, list of images
            shape0, shape1, files = [], [], []  # image and inference shapes, filenames
            for i, im in enumerate(ims):
                f = f'image{i}'  # filename
                if isinstance(im, (str, Path)):  # filename or uri
                    im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im), im
                    im = np.asarray(exif_transpose(im))
                elif isinstance(im, Image.Image):  # PIL Image
                    im, f = np.asarray(exif_transpose(im)), getattr(im, 'filename', f) or f
                files.append(Path(f).with_suffix('.jpg').name)
                if im.shape[0] < 5:  # image in CHW
                    im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
                im = im[..., :3] if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)  # enforce 3ch input
                s = im.shape[:2]  # HWC
                shape0.append(s)  # image shape
                g = max(size) / max(s)  # gain
                shape1.append([int(y * g) for y in s])
                ims[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
            shape1 = [make_divisible(x, self.stride) for x in np.array(shape1).max(0)]  # inf shape
            x = [letterbox(im, shape1, auto=False)[0] for im in ims]  # pad
            x = np.ascontiguousarray(np.array(x).transpose((0, 3, 1, 2)))  # stack and BHWC to BCHW
            x = torch.from_numpy(x).to(p.device).type_as(p) / 255  # uint8 to fp16/32

        with amp.autocast(autocast):
            # Inference
            with dt[1]:
                y = self.model(x, augment=augment)  # forward

            # Post-process
            with dt[2]:
                y = non_max_suppression(y if self.dmb else y[0],
                                        self.conf,
                                        self.iou,
                                        self.classes,
                                        self.agnostic,
                                        self.multi_label,
                                        max_det=self.max_det)  # NMS
                for i in range(n):
                    scale_boxes(shape1, y[i][:, :4], shape0[i])

            return Detections(ims, y, files, dt, self.names, x.shape)


class Detections:
    # YOLOv5 detections class for inference results
    def __init__(self, ims, pred, files, times=(0, 0, 0), names=None, shape=None):
        super().__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in ims]  # normalizations
        self.ims = ims  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.times = times  # profiling times
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple(x.t / self.n * 1E3 for x in times)  # timestamps (ms)
        self.s = tuple(shape)  # inference BCHW shape

    def _run(self, pprint=False, show=False, save=False, crop=False, render=False, labels=True, save_dir=Path('')):
        s, crops = '', []
        for i, (im, pred) in enumerate(zip(self.ims, self.pred)):
            s += f'\nimage {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '  # string
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                s = s.rstrip(', ')
                if show or save or render or crop:
                    annotator = Annotator(im, example=str(self.names))
                    for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            file = save_dir / 'crops' / self.names[int(cls)] / self.files[i] if save else None
                            crops.append({
                                'box': box,
                                'conf': conf,
                                'cls': cls,
                                'label': label,
                                'im': save_one_box(box, im, file=file, save=save)})
                        else:  # all others
                            annotator.box_label(box, label if labels else '', color=colors(cls))
                    im = annotator.im
            else:
                s += '(no detections)'

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if show:
                display(im) if is_notebook() else im.show(self.files[i])
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                if i == self.n - 1:
                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
            if render:
                self.ims[i] = np.asarray(im)
        if pprint:
            s = s.lstrip('\n')
            return f'{s}\nSpeed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {self.s}' % self.t
        if crop:
            if save:
                LOGGER.info(f'Saved results to {save_dir}\n')
            return crops

    @TryExcept('Showing images is not supported in this environment')
    def show(self, labels=True):
        self._run(show=True, labels=labels)  # show results

    def save(self, labels=True, save_dir='runs/detect/exp', exist_ok=False):
        save_dir = increment_path(save_dir, exist_ok, mkdir=True)  # increment save_dir
        self._run(save=True, labels=labels, save_dir=save_dir)  # save results

    def crop(self, save=True, save_dir='runs/detect/exp', exist_ok=False):
        save_dir = increment_path(save_dir, exist_ok, mkdir=True) if save else None
        return self._run(crop=True, save=save, save_dir=save_dir)  # crop results

    def render(self, labels=True):
        self._run(render=True, labels=labels)  # render results
        return self.ims

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        r = range(self.n)  # iterable
        x = [Detections([self.ims[i]], [self.pred[i]], [self.files[i]], self.times, self.names, self.s) for i in r]
        # for d in x:
        #    for k in ['ims', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
        #        setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def print(self):
        LOGGER.info(self.__str__())

    def __len__(self):  # override len(results)
        return self.n

    def __str__(self):  # override print(results)
        return self._run(pprint=True)  # print results

    def __repr__(self):
        return f'YOLOv5 {self.__class__} instance\n' + self.__str__()


class Proto(nn.Module):
    # YOLOv5 mask Proto module for segmentation models
    def __init__(self, c1, c_=256, c2=32):  # ch_in, number of protos, number of masks
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class Classify(nn.Module):
    # YOLOv5 classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, autopad(k, p), g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        return self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))


class DCNv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=1, dilation=1, groups=1, deformable_groups=1):
        super(DCNv2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        out_channels_offset_mask = (self.deformable_groups * 3 *
                                    self.kernel_size[0] * self.kernel_size[1])
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            out_channels_offset_mask,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = Conv.default_act
        self.reset_parameters()

    def forward(self, x):
        offset_mask = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        x = torch.ops.torchvision.deform_conv2d(
            x,
            self.weight,
            offset,
            mask,
            self.bias,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            self.groups,
            self.deformable_groups,
            True
        )
        x = self.bn(x)
        x = self.act(x)
        return x

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        std = 1. / math.sqrt(n)
        self.weight.data.uniform_(-std, std)
        self.bias.data.zero_()
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()


class Bottleneck_DCN(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = DCNv2(c_, c2, 3, 1, groups=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3_DCN(C3):
    #  带有DCNv2的C3模块
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_DCN(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class SEPC_12(nn.Module): #输入为 20*20*512 小图像深通道应该保持深度
    def __init__(self, in_channels, out_channels, e = 0.5):
        super().__init__()
        self.out_channels = out_channels
        c_ = int(out_channels * e)
        self.conv1 = nn.Conv2d(in_channels, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False) #降维 + 残差


        self.conv2_0 = nn.Conv2d(c_, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
        self.conv2_1 = nn.Conv2d(c_, c_, kernel_size=3, padding=1, stride=1, groups=1, bias=False)
        self.conv2_2 = ODConv2d_3rd(c_, c_, kernel_size=3, stride=1, padding=1)
        self.cat = torch.cat
        self.silu = nn.SiLU()
        self.interpolate = F.interpolate
        self.split = torch.split
        self.bn = nn.BatchNorm2d(128)

    def iBN(self, fms, bn):
        sizes = [p.shape[2:] for p in fms]
        n, c = fms[0].shape[0], fms[0].shape[1]
        fm = torch.cat([p.view(n, c, 1, -1) for p in fms], dim=-1)
        fm = self.bn(fm)
        fm = self.split(fm, [s[0] * s[1] for s in sizes], dim=-1)
        return [p.view(n, c, s[0], s[1]) for p, s in zip(fm, sizes)]


    def forward(self, x):
        image = [None] * 2
        x = self.conv1(x)
        # print(x.shape)
        image[0] = x + self.conv2_2(x)
        image[1] = self.conv2_1(x) + self.conv2_2(x) + x
        out = self.iBN(image, nn.BatchNorm2d(self.out_channels))
        out = [self.silu(x.clone()) for x in out]
        out = self.cat([out[0],out[1]], 1)
        # print(out.shape)
        return out

class SEPC_24(nn.Module):
    def __init__(self, in_channels, out_channels, e = 0.5):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        c_ = int(out_channels * e)
        self.conv1 = nn.Conv2d(in_channels, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False) #降维 + 残差
        self.conv1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 残差
        self.conv1_3 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 降维

        self.conv2_0 = nn.Conv2d(c_, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
        self.conv2_1 = nn.Conv2d(c_, c_, kernel_size=3, padding=1, stride=1, groups=1, bias=False)
        self.conv2_2 = ODConv2d_3rd(c_, c_, kernel_size=3, stride=1, padding=1)
        self.conv2_3 = ODConv2d_3rd(c_, c_, kernel_size=5, stride=1, padding=2)
        self.cat = torch.cat
        self.silu = nn.SiLU()
        self.interpolate = F.interpolate
        self.split = torch.split
        self.bn = nn.BatchNorm2d(128)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def iBN(self, fms, bn):
        sizes = [p.shape[2:] for p in fms]
        n, c = fms[0].shape[0], fms[0].shape[1]
        fm = torch.cat([p.view(n, c, 1, -1) for p in fms], dim=-1)
        fm = self.bn(fm)
        fm = self.split(fm, [s[0] * s[1] for s in sizes], dim=-1)
        return [p.view(n, c, s[0], s[1]) for p, s in zip(fm, sizes)]

    def forward(self, x):
        image = [None] * 4
        x = self.conv1(x)
        # print(x.shape)
        image[0] = x + self.conv2_2(x) + self.conv2_3(x)
        image[1] = self.conv2_1(x) + self.conv2_2(x) + self.conv2_3(x) + x + self.conv2_0(x)
        image[2] = self.conv2_0(x) + self.conv2_1(x) + x
        image[3] = x
        out = self.iBN(image, nn.BatchNorm2d(self.out_channels))
        out = [self.silu(x.clone()) for x in out]
        # print(out[0].shape)
        # print(out[1].shape)
        out = self.cat([out[0],out[1],out[2],out[3]], 1)
        out = self.conv1_3(out)
        # print(out.shape)
        out = self.bn1(out)
        # print(out.shape)
        out = self.silu(out)
        # print(out.shape)
        return out

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
 
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None
 
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
 
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
 
class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid_(x_out) 
        return x * scale
 
class SEPC_T(nn.Module):
    def __init__(self, in_channels, out_channels, e = 0.5, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(SEPC_T, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
        self.out_channels = out_channels
        self.in_channels = in_channels
        c_ = int(out_channels * e)
        self.conv1 = nn.Conv2d(in_channels, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False) #降维 + 残差
        self.conv1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 残差
        self.conv1_3 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 降维

        self.conv2_0 = nn.Conv2d(c_, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
        self.conv2_1 = nn.Conv2d(c_, c_, kernel_size=3, padding=1, stride=1, groups=1, bias=False)
        self.conv2_2 = ODConv2d_3rd(c_, c_, kernel_size=3, stride=1, padding=1)
        self.conv2_3 = ODConv2d_3rd(c_, c_, kernel_size=5, stride=1, padding=2)
        self.cat = torch.cat
        self.silu = nn.SiLU()
        self.interpolate = F.interpolate
        self.split = torch.split
        self.bn = nn.BatchNorm2d(128)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def iBN(self, fms, bn):
        sizes = [p.shape[2:] for p in fms]
        n, c = fms[0].shape[0], fms[0].shape[1]
        fm = torch.cat([p.view(n, c, 1, -1) for p in fms], dim=-1)
        fm = self.bn(fm)
        fm = self.split(fm, [s[0] * s[1] for s in sizes], dim=-1)
        return [p.view(n, c, s[0], s[1]) for p, s in zip(fm, sizes)]
    
    def forward(self, x):
        image = [None] * 5
        image1 = [None] * 4
        x = self.conv1(x)
        # print(x.shape)
        image[0] = self.conv2_0(x)
        # print(image[0].shape)
        image[1] = self.conv2_1(x)
        # print(image[1].shape)
        image[2] = self.conv2_2(x)
        # print(image[2].shape)
        image[3] = self.conv2_3(x)
        # print(image[3].shape)
        image[4] = x
        # print(image[4].shape)
        x_out = self.SpatialGate(image[3])
        x_perm1 = image[1].permute(0,2,1,3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        # print(x_out1.shape)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        # print(x_out11.shape)
        x_perm2 = image[2].permute(0,3,2,1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        # print(x_out2.shape)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()
        # print(x_out21.shape)
        image1[0] = x + x_out + x_out21
        image1[0] = self.conv2_2(x)
        image1[1] = image[0] + x_out11 + x_out21 + x_out
        image1[1] = self.conv2_2(x)
        image1[2] = x + x_out11 + image[0]
        image1[2] = self.conv2_2(x)
        image1[3] = x
        image1[3] = self.conv2_2(x)
        out = self.iBN(image1, nn.BatchNorm2d(self.out_channels))
        out = [self.silu(x.clone()) for x in out]
        # print(out[0].shape)
        # print(out[1].shape)
        out = self.cat([out[0],out[1],out[2],out[3]], 1)
        out = self.conv1_3(out)
        # print(out.shape)
        out = self.bn1(out)
        # print(out.shape)
        out = self.silu(out)
        # print(out.shape)
        return out
        # x_perm1 = x.permute(0,2,1,3).contiguous()
        # x_out1 = self.ChannelGateH(x_perm1)
        # x_out11 = x_out1.permute(0,2,1,3).contiguous()
        # x_perm2 = x.permute(0,3,2,1).contiguous()
        # x_out2 = self.ChannelGateW(x_perm2)
        # x_out21 = x_out2.permute(0,3,2,1).contiguous()
        # if not self.no_spatial:
        #     x_out = self.SpatialGate(x)
        #     x_out = (1/3)*(x_out + x_out11 + x_out21)
        # else:
        #     x_out = (1/2)*(x_out11 + x_out21)
        # return x_out




from models.akconv import AKConv
class SEPC_T2(nn.Module):
    def __init__(self, in_channels, out_channels, e = 0.5, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(SEPC_T2, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
        self.out_channels = out_channels
        self.in_channels = in_channels
        c_ = int(out_channels * e)
        self.conv1 = nn.Conv2d(in_channels, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False) #降维 + 残差
        self.conv1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 残差
        self.conv1_3 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 降维

        self.conv2_0 = nn.Conv2d(c_, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
        self.conv2_1 = nn.Conv2d(c_, c_, kernel_size=3, padding=1, stride=1, groups=1, bias=False)
        self.conv2_2 = AKConv(c_, c_, 3, stride=1, bias=None)
        self.conv2_3 = AKConv(c_, c_, 5, stride=1, bias=None)
        self.cat = torch.cat
        self.silu = nn.SiLU()
        self.interpolate = F.interpolate
        self.split = torch.split
        self.bn = nn.BatchNorm2d(128)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def iBN(self, fms, bn):
        sizes = [p.shape[2:] for p in fms]
        n, c = fms[0].shape[0], fms[0].shape[1]
        fm = torch.cat([p.view(n, c, 1, -1) for p in fms], dim=-1)
        fm = self.bn(fm)
        fm = self.split(fm, [s[0] * s[1] for s in sizes], dim=-1)
        return [p.view(n, c, s[0], s[1]) for p, s in zip(fm, sizes)]
    
    def forward(self, x):
        image = [None] * 5
        image1 = [None] * 4
        x = self.conv1(x)
        # print(x.shape)
        image[0] = self.conv2_0(x)
        # print(image[0].shape)
        image[1] = self.conv2_1(x)
        # print(image[1].shape)
        image[2] = self.conv2_2(x)
        # print(image[2].shape)
        image[3] = self.conv2_3(x)
        # print(image[3].shape)
        image[4] = x
        # print(image[4].shape)
        x_out = self.SpatialGate(image[3])
        x_perm1 = image[1].permute(0,2,1,3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        # print(x_out1.shape)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        # print(x_out11.shape)
        x_perm2 = image[2].permute(0,3,2,1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        # print(x_out2.shape)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()
        # print(x_out21.shape)
        image1[0] = x + x_out + x_out21
        image1[0] = self.conv2_2(x)
        image1[1] = image[0] + x_out11 + x_out21 + x_out
        image1[1] = self.conv2_2(x)
        image1[2] = x + x_out11 + image[0]
        image1[2] = self.conv2_2(x)
        image1[3] = x
        image1[3] = self.conv2_2(x)
        out = self.iBN(image1, nn.BatchNorm2d(self.out_channels))
        out = [self.silu(x.clone()) for x in out]
        # print(out[0].shape)
        # print(out[1].shape)
        out = self.cat([out[0],out[1],out[2],out[3]], 1)
        # print(out.shape)
        out = self.conv1_3(out)
        # print(out.shape)
        out = self.bn1(out)
        # print(out.shape)
        out = self.silu(out)
        # print(out.shape)
        return out
        # x_perm1 = x.permute(0,2,1,3).contiguous()
        # x_out1 = self.ChannelGateH(x_perm1)
        # x_out11 = x_out1.permute(0,2,1,3).contiguous()
        # x_perm2 = x.permute(0,3,2,1).contiguous()
        # x_out2 = self.ChannelGateW(x_perm2)
        # x_out21 = x_out2.permute(0,3,2,1).contiguous()
        # if not self.no_spatial:
        #     x_out = self.SpatialGate(x)
        #     x_out = (1/3)*(x_out + x_out11 + x_out21)
        # else:
        #     x_out = (1/2)*(x_out11 + x_out21)
        # return x_out






from models.akconv import AKConv
class SEPC_AK(nn.Module):
    def __init__(self, in_channels, out_channels, e = 0.5, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(SEPC_AK, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
        self.out_channels = out_channels
        self.in_channels = in_channels
        c_ = int(out_channels * e)
        self.conv1 = nn.Conv2d(in_channels, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False) #降维 + 残差
        self.conv1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 残差
        self.conv1_3 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 降维

        self.conv2_0 = nn.Conv2d(c_, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
        self.conv2_1 = nn.Conv2d(c_, c_, kernel_size=3, padding=1, stride=1, groups=1, bias=False)
        self.conv2_2 = AKConv(c_, c_, 3, stride=1, bias=None)
        self.conv2_3 = AKConv(c_, c_, 5, stride=1, bias=None)
        self.cat = torch.cat
        self.silu = nn.SiLU()
        self.interpolate = F.interpolate
        self.split = torch.split
        self.bn = nn.BatchNorm2d(128)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def iBN(self, fms, bn):
        sizes = [p.shape[2:] for p in fms]
        n, c = fms[0].shape[0], fms[0].shape[1]
        fm = torch.cat([p.view(n, c, 1, -1) for p in fms], dim=-1)
        fm = self.bn(fm)
        fm = self.split(fm, [s[0] * s[1] for s in sizes], dim=-1)
        return [p.view(n, c, s[0], s[1]) for p, s in zip(fm, sizes)]
    
    def forward(self, x):
        image = [None] * 5
        image1 = [None] * 4
        x = self.conv1(x)
        # print(x.shape)
        image[0] = self.conv2_0(x)
        # print(image[0].shape)
        image[1] = self.conv2_1(x)
        # print(image[1].shape)
        image[2] = self.conv2_2(x)
        # print(image[2].shape)
        image[3] = self.conv2_3(x)
        # print(image[3].shape)
        image[4] = x
        # print(image[4].shape)
        x_out = self.SpatialGate(image[3])
        x_perm1 = image[1].permute(0,2,1,3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        # print(x_out1.shape)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        # print(x_out11.shape)
        x_perm2 = image[2].permute(0,3,2,1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        # print(x_out2.shape)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()
        # print(x_out21.shape)
        image1[0] = x + x_out + x_out21
        # image1[0] = self.conv2_2(x)
        image1[1] = image[0] + x_out11 + x_out21 + x_out
        # image1[1] = self.conv2_2(x)
        image1[2] = x + x_out11 + image[0]
        # image1[2] = self.conv2_2(x)
        image1[3] = x
        # image1[3] = self.conv2_2(x)
        out = self.iBN(image1, nn.BatchNorm2d(self.out_channels))
        out = [self.silu(x.clone()) for x in out]
        # print(out[0].shape)
        # print(out[1].shape)
        out = self.cat([out[0],out[1],out[2],out[3]], 1)
        # print(out.shape)
        out = self.conv1_3(out)
        # print(out.shape)
        out = self.bn1(out)
        # print(out.shape)
        out = self.silu(out)
        # print(out.shape)
        return out
        # x_perm1 = x.permute(0,2,1,3).contiguous()
        # x_out1 = self.ChannelGateH(x_perm1)
        # x_out11 = x_out1.permute(0,2,1,3).contiguous()
        # x_perm2 = x.permute(0,3,2,1).contiguous()
        # x_out2 = self.ChannelGateW(x_perm2)
        # x_out21 = x_out2.permute(0,3,2,1).contiguous()
        # if not self.no_spatial:
        #     x_out = self.SpatialGate(x)
        #     x_out = (1/3)*(x_out + x_out11 + x_out21)
        # else:
        #     x_out = (1/2)*(x_out11 + x_out21)
        # return x_out
    


from models.akconv import AKConv
class SEPC_AKsmall(nn.Module):
    def __init__(self, in_channels, out_channels, e = 0.5, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(SEPC_AK, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
        self.out_channels = out_channels
        self.in_channels = in_channels
        c_ = int(out_channels * e)
        self.conv1 = nn.Conv2d(in_channels, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False) #降维 + 残差
        self.conv1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 残差
        self.conv1_3 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 降维

        self.conv2_0 = nn.Conv2d(c_, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
        self.conv2_1 = nn.Conv2d(c_, c_, kernel_size=3, padding=1, stride=1, groups=1, bias=False)
        self.conv2_2 = AKConv(c_, c_, 3, stride=1, bias=None)
        self.conv2_3 = AKConv(c_, c_, 5, stride=1, bias=None)
        self.cat = torch.cat
        self.silu = nn.SiLU()
        self.interpolate = F.interpolate
        self.split = torch.split
        self.bn = nn.BatchNorm2d(64)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def iBN(self, fms, bn):
        sizes = [p.shape[2:] for p in fms]
        n, c = fms[0].shape[0], fms[0].shape[1]
        fm = torch.cat([p.view(n, c, 1, -1) for p in fms], dim=-1)
        fm = self.bn(fm)
        fm = self.split(fm, [s[0] * s[1] for s in sizes], dim=-1)
        return [p.view(n, c, s[0], s[1]) for p, s in zip(fm, sizes)]
    
    def forward(self, x):
        image = [None] * 5
        image1 = [None] * 4
        x = self.conv1(x)
        # print(x.shape)
        image[0] = self.conv2_0(x)
        # print(image[0].shape)
        image[1] = self.conv2_1(x)
        # print(image[1].shape)
        image[2] = self.conv2_2(x)
        # print(image[2].shape)
        image[3] = self.conv2_3(x)
        # print(image[3].shape)
        image[4] = x
        # print(image[4].shape)
        x_out = self.SpatialGate(image[3])
        x_perm1 = image[1].permute(0,2,1,3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        # print(x_out1.shape)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        # print(x_out11.shape)
        x_perm2 = image[2].permute(0,3,2,1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        # print(x_out2.shape)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()
        # print(x_out21.shape)
        image1[0] = x + x_out + x_out21
        # image1[0] = self.conv2_2(x)
        image1[1] = image[0] + x_out11 + x_out21 + x_out
        # image1[1] = self.conv2_2(x)
        image1[2] = x + x_out11 + image[0]
        # image1[2] = self.conv2_2(x)
        image1[3] = x
        # image1[3] = self.conv2_2(x)
        out = self.iBN(image1, nn.BatchNorm2d(self.out_channels))
        out = [self.silu(x.clone()) for x in out]
        # print(out[0].shape)
        # print(out[1].shape)
        out = self.cat([out[0],out[1],out[2],out[3]], 1)
        # print(out.shape)
        out = self.conv1_3(out)
        # print(out.shape)
        out = self.bn1(out)
        # print(out.shape)
        out = self.silu(out)
        # print(out.shape)
        return out
        # x_perm1 = x.permute(0,2,1,3).contiguous()
        # x_out1 = self.ChannelGateH(x_perm1)
        # x_out11 = x_out1.permute(0,2,1,3).contiguous()
        # x_perm2 = x.permute(0,3,2,1).contiguous()
        # x_out2 = self.ChannelGateW(x_perm2)
        # x_out21 = x_out2.permute(0,3,2,1).contiguous()
        # if not self.no_spatial:
        #     x_out = self.SpatialGate(x)
        #     x_out = (1/3)*(x_out + x_out11 + x_out21)
        # else:
        #     x_out = (1/2)*(x_out11 + x_out21)
        # return x_out




from models.akconv import AKConv
class SEPC_AK_avg(nn.Module):
    def __init__(self, in_channels, out_channels, e = 0.5, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(SEPC_AK_avg, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
        self.out_channels = out_channels
        self.in_channels = in_channels
        c_ = int(out_channels * e)
        self.conv1 = nn.Conv2d(in_channels, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False) #降维 + 残差
        self.conv1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 残差
        self.conv1_3 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 降维

        self.conv2_0 = nn.Conv2d(c_, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
        self.conv2_1 = nn.Conv2d(c_, c_, kernel_size=3, padding=1, stride=1, groups=1, bias=False)
        self.conv2_2 = AKConv(c_, c_, 3, stride=1, bias=None)
        self.conv2_3 = AKConv(c_, c_, 5, stride=1, bias=None)
        self.cat = torch.cat
        self.silu = nn.SiLU()
        self.interpolate = F.interpolate
        self.split = torch.split
        self.bn = nn.BatchNorm2d(128)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def iBN(self, fms, bn):
        sizes = [p.shape[2:] for p in fms]
        n, c = fms[0].shape[0], fms[0].shape[1]
        fm = torch.cat([p.view(n, c, 1, -1) for p in fms], dim=-1)
        fm = self.bn(fm)
        fm = self.split(fm, [s[0] * s[1] for s in sizes], dim=-1)
        return [p.view(n, c, s[0], s[1]) for p, s in zip(fm, sizes)]
    
    def forward(self, x):
        image = [None] * 5
        image1 = [None] * 4
        x = self.conv1(x)
        # print(x.shape)
        image[0] = self.conv2_0(x)
        # print(image[0].shape)
        image[1] = self.conv2_1(x)
        # print(image[1].shape)
        image[2] = self.conv2_2(x)
        # print(image[2].shape)
        image[3] = self.conv2_3(x)
        # print(image[3].shape)
        image[4] = x
        # print(image[4].shape)
        x_out = self.SpatialGate(image[3])
        x_perm1 = image[1].permute(0,2,1,3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        # print(x_out1.shape)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        # print(x_out11.shape)
        x_perm2 = image[2].permute(0,3,2,1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        # print(x_out2.shape)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()
        # print(x_out21.shape)
        image1[0] = 1/3 * (x + x_out + x_out21)
        # image1[0] = self.conv2_2(x)
        image1[1] = 1/3 * (x_out11 + x_out21 + x_out)
        # image1[1] = self.conv2_2(x)
        image1[2] = 1/3 * (x + x_out11 + x_out)
        # image1[2] = self.conv2_2(x)
        image1[3] = x
        # image1[3] = self.conv2_2(x)
        out = self.iBN(image1, nn.BatchNorm2d(self.out_channels))
        out = [self.silu(x.clone()) for x in out]
        # print(out[0].shape)
        # print(out[1].shape)
        out = self.cat([out[0],out[1],out[2],out[3]], 1)
        # print(out.shape)
        out = self.conv1_3(out)
        # print(out.shape)
        out = self.bn1(out)
        # print(out.shape)
        out = self.silu(out)
        # print(out.shape)
        return out
        # x_perm1 = x.permute(0,2,1,3).contiguous()
        # x_out1 = self.ChannelGateH(x_perm1)
        # x_out11 = x_out1.permute(0,2,1,3).contiguous()
        # x_perm2 = x.permute(0,3,2,1).contiguous()
        # x_out2 = self.ChannelGateW(x_perm2)
        # x_out21 = x_out2.permute(0,3,2,1).contiguous()
        # if not self.no_spatial:
        #     x_out = self.SpatialGate(x)
        #     x_out = (1/3)*(x_out + x_out11 + x_out21)
        # else:
        #     x_out = (1/2)*(x_out11 + x_out21)
        # return x_out




class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out) 
        return x * scale

class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.hw = AttentionGate()
    def forward(self, x):
        x_perm1 = x.permute(0,2,1,3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        x_perm2 = x.permute(0,3,2,1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1/3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1/2 * (x_out11 + x_out21)
        return x_out









class SEPC_DSC_new(nn.Module):
    def __init__(self, in_channels, out_channels, e = 0.5, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(SEPC_DSC_new, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
        self.out_channels = out_channels
        self.in_channels = in_channels
        c_ = int(out_channels * e)
        self.conv1 = nn.Conv2d(in_channels, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False) #降维 + 残差
        self.conv1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 残差
        self.conv1_3 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 降维

        self.conv2_0 = nn.Conv2d(c_, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
        self.conv2_1 = nn.Conv2d(c_, c_, kernel_size=3, padding=1, stride=1, groups=1, bias=False)
        self.conv2_2 = DSConv(c_, c_, kernel_size=3, extend_scope=1, morph=1, if_offset=True)
        self.conv2_3 = AKConv(c_, c_, 3, stride=1, bias=None)
        self.tri = TripletAttention()
        self.cat = torch.cat
        self.silu = nn.SiLU()
        self.interpolate = F.interpolate
        self.split = torch.split
        self.bn = nn.BatchNorm2d(128)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def iBN(self, fms, bn):
        sizes = [p.shape[2:] for p in fms]
        n, c = fms[0].shape[0], fms[0].shape[1]
        fm = torch.cat([p.view(n, c, 1, -1) for p in fms], dim=-1)
        fm = self.bn(fm)
        fm = self.split(fm, [s[0] * s[1] for s in sizes], dim=-1)
        return [p.view(n, c, s[0], s[1]) for p, s in zip(fm, sizes)]
    
    def forward(self, x):
        image = [None] * 6
        image1 = [None] * 4
        x = self.conv1(x)
        # print(x.shape)
        image[0] = self.conv2_0(x)
        # print(image[0].shape)
        image[1] = self.conv2_1(x)
        # print(image[1].shape)
        image[2] = self.conv2_1(self.conv2_1(x))
        # print(image[2].shape)
        image[3] = self.conv2_2(x)
        # print(image[2].shape)
        image[4] = self.conv2_2(self.conv2_2(x))
        # print(image[3].shape)
        image[5] = x
        # print(image[4].shape)
        image1[0] = image[0] + image[1] + image[2] + image[5] 
        # image1[0] = self.conv2_2(x)
        image1[1] = image[1] + image[2] + image[3] + image[4]
        # image1[1] = self.conv2_2(x)
        image1[2] = image[0] + image[3] + image[4] + image[5]
        # image1[2] = self.conv2_2(x)
        image1[3] = image[5]
        # image1[3] = self.conv2_2(x)
        image1[0] = self.tri(image1[0])
        image1[1] = self.tri(image1[1])
        image1[2] = self.tri(image1[2])
        image1[3] = self.tri(image1[3])
        out = self.iBN(image1, nn.BatchNorm2d(self.out_channels))
        out = [self.silu(x.clone()) for x in out]
        # print(out[0].shape)
        # print(out[1].shape)
        out = self.cat([out[0],out[1],out[2],out[3]], 1)
        # print(out.shape)
        # out = self.tri(out)
        out = self.conv1_3(out)
        # print(out.shape)
        out = self.bn1(out)
        # print(out.shape)
        out = self.silu(out)
        # print(out.shape)
        return out
        # x_perm1 = x.permute(0,2,1,3).contiguous()
        # x_out1 = self.ChannelGateH(x_perm1)
        # x_out11 = x_out1.permute(0,2,1,3).contiguous()
        # x_perm2 = x.permute(0,3,2,1).contiguous()
        # x_out2 = self.ChannelGateW(x_perm2)
        # x_out21 = x_out2.permute(0,3,2,1).contiguous()
        # if not self.no_spatial:
        #     x_out = self.SpatialGate(x)
        #     x_out = (1/3)*(x_out + x_out11 + x_out21)
        # else:
        #     x_out = (1/2)*(x_out11 + x_out21)
        # return x_out


class SEPC_DSC122(nn.Module):
    def __init__(self, in_channels, out_channels, e = 0.5, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(SEPC_DSC122, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
        self.out_channels = out_channels
        self.in_channels = in_channels
        c_ = int(out_channels * e)
        self.conv1 = nn.Conv2d(in_channels, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False) #降维 + 残差
        self.conv1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 残差
        self.conv1_3 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 降维

        self.conv2_0 = nn.Conv2d(c_, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
        self.conv2_1 = nn.Conv2d(c_, c_, kernel_size=3, padding=1, stride=1, groups=1, bias=False)
        self.conv2_2 = DSConv(c_, c_, kernel_size=3, extend_scope=1, morph=1, if_offset=True)
        self.conv2_3 = AKConv(c_, c_, 3, stride=1, bias=None)
        self.tri = TripletAttention()
        self.cat = torch.cat
        self.silu = nn.SiLU()
        self.interpolate = F.interpolate
        self.split = torch.split
        self.bn = nn.BatchNorm2d(128)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def iBN(self, fms, bn):
        sizes = [p.shape[2:] for p in fms]
        n, c = fms[0].shape[0], fms[0].shape[1]
        fm = torch.cat([p.view(n, c, 1, -1) for p in fms], dim=-1)
        fm = self.bn(fm)
        fm = self.split(fm, [s[0] * s[1] for s in sizes], dim=-1)
        return [p.view(n, c, s[0], s[1]) for p, s in zip(fm, sizes)]
    
    def forward(self, x):
        image = [None] * 6
        image1 = [None] * 4
        x = self.conv1(x)
        # print(x.shape)
        image[0] = self.conv2_0(x)
        # print(image[0].shape)
        image[1] = self.conv2_1(x)
        # print(image[1].shape)
        image[2] = self.conv2_1(self.conv2_1(x))
        # print(image[2].shape)
        image[3] = self.conv2_2(x)
        # print(image[2].shape)
        image[4] = self.conv2_2(self.conv2_2(x))
        # print(image[3].shape)
        image[5] = x
        # print(image[4].shape)
        image1[0] = image[0] + image[1] + image[2] + image[5] 
        # image1[0] = self.conv2_2(x)
        image1[1] = image[1] + image[2] + image[3] + image[4]
        # image1[1] = self.conv2_2(x)
        image1[2] = image[0] + image[3] + image[4] + image[5]
        # image1[2] = self.conv2_2(x)
        image1[3] = image[5]
        # image1[3] = self.conv2_2(x)

        image1[1] = self.SpatialGate(image1[1])
        x_perm1 = image1[2].permute(0,2,1,3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        # print(x_out1.shape)
        image1[2] = x_out1.permute(0,2,1,3).contiguous()
        # print(x_out11.shape)
        x_perm2 = image1[0].permute(0,3,2,1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        # print(x_out2.shape)
        image1[0] = x_out2.permute(0,3,2,1).contiguous()
        # print(x_out21.shape)
        image1[3] = image[5]
        out = self.iBN(image1, nn.BatchNorm2d(self.out_channels))
        out = [self.silu(x.clone()) for x in out]
        # print(out[0].shape)
        # print(out[1].shape)
        out = self.cat([out[0],out[1],out[2],out[3]], 1)
        # print(out.shape)
        # out = self.tri(out)
        out = self.conv1_3(out)
        # print(out.shape)
        out = self.bn1(out)
        # print(out.shape)
        out = self.silu(out)
        # print(out.shape)
        return out
        # x_perm1 = x.permute(0,2,1,3).contiguous()
        # x_out1 = self.ChannelGateH(x_perm1)
        # x_out11 = x_out1.permute(0,2,1,3).contiguous()
        # x_perm2 = x.permute(0,3,2,1).contiguous()
        # x_out2 = self.ChannelGateW(x_perm2)
        # x_out21 = x_out2.permute(0,3,2,1).contiguous()
        # if not self.no_spatial:
        #     x_out = self.SpatialGate(x)
        #     x_out = (1/3)*(x_out + x_out11 + x_out21)
        # else:
        #     x_out = (1/2)*(x_out11 + x_out21)
        # return x_out


from models.akconv import AKConv
class SEPC_AK122(nn.Module):
    def __init__(self, in_channels, out_channels, e = 0.5, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(SEPC_AK122, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
        self.out_channels = out_channels
        self.in_channels = in_channels
        c_ = int(out_channels * e)
        self.conv1 = nn.Conv2d(in_channels, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False) #降维 + 残差
        self.conv1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 残差
        self.conv1_3 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 降维

        self.conv2_0 = nn.Conv2d(c_, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
        self.conv2_1 = nn.Conv2d(c_, c_, kernel_size=3, padding=1, stride=1, groups=1, bias=False)
        self.conv2_2 = AKConv(c_, c_, 3, stride=1, bias=None)
        self.conv2_3 = AKConv(c_, c_, 3, stride=1, bias=None)
        self.tri = TripletAttention()
        self.cat = torch.cat
        self.silu = nn.SiLU()
        self.interpolate = F.interpolate
        self.split = torch.split
        self.bn = nn.BatchNorm2d(128)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def iBN(self, fms, bn):
        sizes = [p.shape[2:] for p in fms]
        n, c = fms[0].shape[0], fms[0].shape[1]
        fm = torch.cat([p.view(n, c, 1, -1) for p in fms], dim=-1)
        fm = self.bn(fm)
        fm = self.split(fm, [s[0] * s[1] for s in sizes], dim=-1)
        return [p.view(n, c, s[0], s[1]) for p, s in zip(fm, sizes)]
    
    def forward(self, x):
        image = [None] * 6
        image1 = [None] * 4
        x = self.conv1(x)
        # print(x.shape)
        image[0] = self.conv2_0(x)
        # print(image[0].shape)
        image[1] = self.conv2_1(x)
        # print(image[1].shape)
        image[2] = self.conv2_1(self.conv2_1(x))
        # print(image[2].shape)
        image[3] = self.conv2_2(x)
        # print(image[2].shape)
        image[4] = self.conv2_2(self.conv2_2(x))
        # print(image[3].shape)
        image[5] = x
        # print(image[4].shape)
        image1[0] = image[0] + image[1] + image[2] + image[5] 
        # image1[0] = self.conv2_2(x)
        image1[1] = image[1] + image[2] + image[3] + image[4]
        # image1[1] = self.conv2_2(x)
        image1[2] = image[0] + image[3] + image[4] + image[5]
        # image1[2] = self.conv2_2(x)
        image1[3] = image[5]
        # image1[3] = self.conv2_2(x)
        # image1[0] = self.tri(image1[0])
        # image1[1] = self.tri(image1[1])
        # image1[2] = self.tri(image1[2])
        # image1[3] = self.tri(image1[3])
        
        image1[1] = self.SpatialGate(image1[1])
        x_perm1 = image1[2].permute(0,2,1,3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        # print(x_out1.shape)
        image1[2] = x_out1.permute(0,2,1,3).contiguous()
        # print(x_out11.shape)
        x_perm2 = image1[0].permute(0,3,2,1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        # print(x_out2.shape)
        image1[0] = x_out2.permute(0,3,2,1).contiguous()
        # print(x_out21.shape)
        image1[3] = image[5]

        out = self.iBN(image1, nn.BatchNorm2d(self.out_channels))
        out = [self.silu(x.clone()) for x in out]
        # print(out[0].shape)
        # print(out[1].shape)
        out = self.cat([out[0],out[1],out[2],out[3]], 1)
        # print(out.shape)
        # out = self.tri(out)
        out = self.conv1_3(out)
        # print(out.shape)
        out = self.bn1(out)
        # print(out.shape)
        out = self.silu(out)
        # print(out.shape)
        return out
        # x_perm1 = x.permute(0,2,1,3).contiguous()
        # x_out1 = self.ChannelGateH(x_perm1)
        # x_out11 = x_out1.permute(0,2,1,3).contiguous()
        # x_perm2 = x.permute(0,3,2,1).contiguous()
        # x_out2 = self.ChannelGateW(x_perm2)
        # x_out21 = x_out2.permute(0,3,2,1).contiguous()
        # if not self.no_spatial:
        #     x_out = self.SpatialGate(x)
        #     x_out = (1/3)*(x_out + x_out11 + x_out21)
        # else:
        #     x_out = (1/2)*(x_out11 + x_out21)
        # return x_out


from models.akconv import AKConv
class SEPC_AK122_2(nn.Module):
    def __init__(self, in_channels, out_channels, e = 0.5, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(SEPC_AK122_2, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
        self.out_channels = out_channels
        self.in_channels = in_channels
        c_ = int(out_channels * e)
        self.conv1 = nn.Conv2d(in_channels, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False) #降维 + 残差
        self.conv1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 残差
        self.conv1_3 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 降维

        self.conv2_0 = nn.Conv2d(c_, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
        self.conv2_1 = nn.Conv2d(c_, c_, kernel_size=3, padding=1, stride=1, groups=1, bias=False)
        self.conv2_2 = AKConv(c_, c_, 3, stride=1, bias=None)
        self.conv2_3 = AKConv(c_, c_, 3, stride=1, bias=None)
        self.tri = TripletAttention()
        self.cat = torch.cat
        self.silu = nn.SiLU()
        self.interpolate = F.interpolate
        self.split = torch.split
        self.bn = nn.BatchNorm2d(128)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def iBN(self, fms, bn):
        sizes = [p.shape[2:] for p in fms]
        n, c = fms[0].shape[0], fms[0].shape[1]
        fm = torch.cat([p.view(n, c, 1, -1) for p in fms], dim=-1)
        fm = self.bn(fm)
        fm = self.split(fm, [s[0] * s[1] for s in sizes], dim=-1)
        return [p.view(n, c, s[0], s[1]) for p, s in zip(fm, sizes)]
    
    def forward(self, x):
        image = [None] * 6
        image1 = [None] * 4
        x = self.conv1(x)
        # print(x.shape)
        image[0] = self.conv2_0(x)
        # print(image[0].shape)
        image[1] = self.conv2_1(x)
        # print(image[1].shape)
        image[2] = self.conv2_1(self.conv2_1(x))
        # print(image[2].shape)
        image[3] = self.conv2_2(x)
        # print(image[2].shape)
        image[4] = self.conv2_2(self.conv2_2(x))
        # print(image[3].shape)
        image[5] = x
        # print(image[4].shape)
        image1[0] = image[0] + image[1] + image[2] + image[5] 
        # image1[0] = self.conv2_2(x)
        image1[1] = image[1] + image[2] + image[3] + image[4]
        # image1[1] = self.conv2_2(x)
        image1[2] = image[0] + image[3] + image[4] + image[5]
        # image1[2] = self.conv2_2(x)
        image1[3] = image[5]
        # image1[3] = self.conv2_2(x)
        # image1[0] = self.tri(image1[0])
        # image1[1] = self.tri(image1[1])
        # image1[2] = self.tri(image1[2])
        # image1[3] = self.tri(image1[3])
        
        # image1[1] = self.SpatialGate(image1[1])
        # x_perm1 = image1[2].permute(0,2,1,3).contiguous()
        # x_out1 = self.ChannelGateH(x_perm1)
        # # print(x_out1.shape)
        # image1[2] = x_out1.permute(0,2,1,3).contiguous()
        # # print(x_out11.shape)
        # x_perm2 = image1[0].permute(0,3,2,1).contiguous()
        # x_out2 = self.ChannelGateW(x_perm2)
        # # print(x_out2.shape)
        # image1[0] = x_out2.permute(0,3,2,1).contiguous()
        # # print(x_out21.shape)
        # image1[3] = image[5]

        out = self.iBN(image1, nn.BatchNorm2d(self.out_channels))
        out = [self.silu(x.clone()) for x in out]
        # print(out[0].shape)
        # print(out[1].shape)
        out = self.cat([out[0],out[1],out[2],out[3]], 1)
        # print(out.shape)
        # out = self.tri(out)
        out = self.conv1_3(out)
        # print(out.shape)
        out = self.bn1(out)
        # print(out.shape)
        out = self.silu(out)
        # print(out.shape)
        return out
        # x_perm1 = x.permute(0,2,1,3).contiguous()
        # x_out1 = self.ChannelGateH(x_perm1)
        # x_out11 = x_out1.permute(0,2,1,3).contiguous()
        # x_perm2 = x.permute(0,3,2,1).contiguous()
        # x_out2 = self.ChannelGateW(x_perm2)
        # x_out21 = x_out2.permute(0,3,2,1).contiguous()
        # if not self.no_spatial:
        #     x_out = self.SpatialGate(x)
        #     x_out = (1/3)*(x_out + x_out11 + x_out21)
        # else:
        #     x_out = (1/2)*(x_out11 + x_out21)
        # return x_out



from models.akconv import AKConv
class SEPC_AK122_3(nn.Module):
    def __init__(self, in_channels, out_channels, e = 0.5, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(SEPC_AK122_3, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
        self.out_channels = out_channels
        self.in_channels = in_channels
        c_ = int(out_channels * e)
        self.conv1 = nn.Conv2d(in_channels, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False) #降维 + 残差
        self.conv1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 残差
        self.conv1_3 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 降维

        self.conv2_0 = nn.Conv2d(c_, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
        self.conv2_1 = nn.Conv2d(c_, c_, kernel_size=3, padding=1, stride=1, groups=1, bias=False)
        self.conv2_2 = AKConv(c_, c_, 3, stride=1, bias=None)
        self.conv2_3 = AKConv(c_, c_, 3, stride=1, bias=None)
        self.tri = TripletAttention()
        self.cat = torch.cat
        self.silu = nn.SiLU()
        self.interpolate = F.interpolate
        self.split = torch.split
        self.bn = nn.BatchNorm2d(128)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def iBN(self, fms, bn):
        sizes = [p.shape[2:] for p in fms]
        n, c = fms[0].shape[0], fms[0].shape[1]
        fm = torch.cat([p.view(n, c, 1, -1) for p in fms], dim=-1)
        fm = self.bn(fm)
        fm = self.split(fm, [s[0] * s[1] for s in sizes], dim=-1)
        return [p.view(n, c, s[0], s[1]) for p, s in zip(fm, sizes)]
    
    def forward(self, x):
        image = [None] * 6
        image1 = [None] * 4
        x = self.conv1(x)
        # print(x.shape)
        image[0] = self.conv2_0(x)
        # print(image[0].shape)
        image[1] = self.conv2_1(x)
        # print(image[1].shape)
        image[2] = self.conv2_1(self.conv2_1(x))
        # print(image[2].shape)
        image[3] = self.conv2_2(x)
        # print(image[2].shape)
        image[4] = self.conv2_2(self.conv2_2(x))
        # print(image[3].shape)
        image[5] = x
        # print(image[4].shape)
        
        image[2] = self.SpatialGate(image[2])
        image[4] = self.SpatialGate(image[4])
        x_perm1 = image[1].permute(0,2,1,3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        # print(x_out1.shape)
        image[1] = x_out1.permute(0,2,1,3).contiguous()
        # print(x_out11.shape)
        x_perm2 = image[3].permute(0,3,2,1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        # print(x_out2.shape)
        image[3] = x_out2.permute(0,3,2,1).contiguous()
        # print(x_out21.shape)



        
        image1[0] = image[0] + image[1] + image[2] + image[5] 
        # image1[0] = self.conv2_2(x)
        image1[1] = image[1] + image[2] + image[3] + image[4]
        # image1[1] = self.conv2_2(x)
        image1[2] = image[0] + image[3] + image[4] + image[5]
        # image1[2] = self.conv2_2(x)
        image1[3] = image[5]
        # image1[3] = self.conv2_2(x)
        # image1[0] = self.tri(image1[0])
        # image1[1] = self.tri(image1[1])
        # image1[2] = self.tri(image1[2])
        # image1[3] = self.tri(image1[3])
        

        out = self.iBN(image1, nn.BatchNorm2d(self.out_channels))
        out = [self.silu(x.clone()) for x in out]
        # print(out[0].shape)
        # print(out[1].shape)
        out = self.cat([out[0],out[1],out[2],out[3]], 1)
        # print(out.shape)
        # out = self.tri(out)
        out = self.conv1_3(out)
        # print(out.shape)
        out = self.bn1(out)
        # print(out.shape)
        out = self.silu(out)
        # print(out.shape)
        return out
        # x_perm1 = x.permute(0,2,1,3).contiguous()
        # x_out1 = self.ChannelGateH(x_perm1)
        # x_out11 = x_out1.permute(0,2,1,3).contiguous()
        # x_perm2 = x.permute(0,3,2,1).contiguous()
        # x_out2 = self.ChannelGateW(x_perm2)
        # x_out21 = x_out2.permute(0,3,2,1).contiguous()
        # if not self.no_spatial:
        #     x_out = self.SpatialGate(x)
        #     x_out = (1/3)*(x_out + x_out11 + x_out21)
        # else:
        #     x_out = (1/2)*(x_out11 + x_out21)
        # return x_out





from models.akconv import AKConv
class SEPC_AK122_4(nn.Module):
    def __init__(self, in_channels, out_channels, e = 0.5, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(SEPC_AK122_4, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
        self.out_channels = out_channels
        self.in_channels = in_channels
        c_ = int(out_channels * e)
        self.conv1 = nn.Conv2d(in_channels, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False) #降维 + 残差
        self.conv1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 残差
        self.conv1_3 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 降维

        self.conv2_0 = nn.Conv2d(c_, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
        self.conv2_1 = nn.Conv2d(c_, c_, kernel_size=3, padding=1, stride=1, groups=1, bias=False)
        self.conv2_2 = AKConv(c_, c_, 3, stride=1, bias=None)
        self.conv2_3 = AKConv(c_, c_, 3, stride=1, bias=None)
        self.tri = TripletAttention()
        self.cat = torch.cat
        self.silu = nn.SiLU()
        self.interpolate = F.interpolate
        self.split = torch.split
        self.bn = nn.BatchNorm2d(128)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def iBN(self, fms, bn):
        sizes = [p.shape[2:] for p in fms]
        n, c = fms[0].shape[0], fms[0].shape[1]
        fm = torch.cat([p.view(n, c, 1, -1) for p in fms], dim=-1)
        fm = self.bn(fm)
        fm = self.split(fm, [s[0] * s[1] for s in sizes], dim=-1)
        return [p.view(n, c, s[0], s[1]) for p, s in zip(fm, sizes)]
    
    def forward(self, x):
        image = [None] * 8
        image1 = [None] * 4
        x = self.conv1(x)
        # print(x.shape)
        image[0] = self.conv2_0(x)
        # print(image[0].shape)
        image[1] = self.conv2_1(x)
        # print(image[1].shape)
        image[2] = self.conv2_1(self.conv2_1(x))
        # print(image[2].shape)
        image[3] = self.conv2_2(x)
        # print(image[2].shape)
        image[4] = self.conv2_2(self.conv2_2(x))
        # print(image[3].shape)
        image[5] = x
        image[6] = image[1]
        image[7] = image[3]
        # print(image[4].shape)
        
        image[2] = self.SpatialGate(image[2])
        image[4] = self.SpatialGate(image[4])

        x_perm1 = image[1].permute(0,2,1,3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        # print(x_out1.shape)
        image[1] = x_out1.permute(0,2,1,3).contiguous()
        # print(x_out11.shape)
        x_perm2 = image[6].permute(0,3,2,1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        # print(x_out2.shape)
        image[6] = x_out2.permute(0,3,2,1).contiguous()
        # print(x_out21.shape)

        x_perm1 = image[7].permute(0,2,1,3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        # print(x_out1.shape)
        image[7] = x_out1.permute(0,2,1,3).contiguous()
        # print(x_out11.shape)
        x_perm2 = image[3].permute(0,3,2,1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        # print(x_out2.shape)
        image[3] = x_out2.permute(0,3,2,1).contiguous()
        # print(x_out21.shape)



        
        image1[0] = image[0] + image[1] + image[2] + image[5] + image[6]
        # image1[0] = self.conv2_2(x)
        image1[1] = image[1] + image[2] + image[3] + image[4] + image[6] + image[7]
        # image1[1] = self.conv2_2(x)
        image1[2] = image[0] + image[3] + image[4] + image[5] + image[7]
        # image1[2] = self.conv2_2(x)
        image1[3] = image[5]
        # image1[3] = self.conv2_2(x)
        # image1[0] = self.tri(image1[0])
        # image1[1] = self.tri(image1[1])
        # image1[2] = self.tri(image1[2])
        # image1[3] = self.tri(image1[3])
        

        out = self.iBN(image1, nn.BatchNorm2d(self.out_channels))
        out = [self.silu(x.clone()) for x in out]
        # print(out[0].shape)
        # print(out[1].shape)
        out = self.cat([out[0],out[1],out[2],out[3]], 1)
        # print(out.shape)
        # out = self.tri(out)
        out = self.conv1_3(out)
        # print(out.shape)
        out = self.bn1(out)
        # print(out.shape)
        out = self.silu(out)
        # print(out.shape)
        return out
        # x_perm1 = x.permute(0,2,1,3).contiguous()
        # x_out1 = self.ChannelGateH(x_perm1)
        # x_out11 = x_out1.permute(0,2,1,3).contiguous()
        # x_perm2 = x.permute(0,3,2,1).contiguous()
        # x_out2 = self.ChannelGateW(x_perm2)
        # x_out21 = x_out2.permute(0,3,2,1).contiguous()
        # if not self.no_spatial:
        #     x_out = self.SpatialGate(x)
        #     x_out = (1/3)*(x_out + x_out11 + x_out21)
        # else:
        #     x_out = (1/2)*(x_out11 + x_out21)
        # return x_out






from models.akconv import AKConv
class SEPC_AK122_5(nn.Module):
    def __init__(self, in_channels, out_channels, e = 0.5, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(SEPC_AK122_5, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
        self.out_channels = out_channels
        self.in_channels = in_channels
        c_ = int(out_channels * e)
        self.conv1 = nn.Conv2d(in_channels, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False) #降维 + 残差
        self.conv1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 残差
        self.conv1_3 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 降维

        self.conv2_0 = nn.Conv2d(c_, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
        self.conv2_1 = nn.Conv2d(c_, c_, kernel_size=3, padding=1, stride=1, groups=1, bias=False)
        self.conv2_2 = AKConv(c_, c_, 3, stride=1, bias=None)
        self.conv2_3 = AKConv(c_, c_, 3, stride=1, bias=None)
        self.tri = TripletAttention()
        self.cat = torch.cat
        self.silu = nn.SiLU()
        self.interpolate = F.interpolate
        self.split = torch.split
        self.bn = nn.BatchNorm2d(128)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def iBN(self, fms, bn):
        sizes = [p.shape[2:] for p in fms]
        n, c = fms[0].shape[0], fms[0].shape[1]
        fm = torch.cat([p.view(n, c, 1, -1) for p in fms], dim=-1)
        fm = self.bn(fm)
        fm = self.split(fm, [s[0] * s[1] for s in sizes], dim=-1)
        return [p.view(n, c, s[0], s[1]) for p, s in zip(fm, sizes)]
    
    def forward(self, x):
        image = [None] * 8
        image1 = [None] * 4
        x = self.conv1(x)
        # print(x.shape)
        image[0] = self.conv2_0(x)
        # print(image[0].shape)
        image[1] = self.conv2_1(x)
        # print(image[1].shape)
        image[2] = self.conv2_1(self.conv2_1(x))
        # print(image[2].shape)
        image[3] = self.conv2_2(x)
        # print(image[2].shape)
        image[4] = self.conv2_2(self.conv2_2(x))
        # print(image[3].shape)
        image[5] = x
        image[6] = image[1]
        image[7] = image[3]
        # print(image[4].shape)
        
        image[2] = self.SpatialGate(image[2])
        image[4] = self.SpatialGate(image[4])

        x_perm1 = image[1].permute(0,2,1,3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        # print(x_out1.shape)
        image[1] = x_out1.permute(0,2,1,3).contiguous()
        # print(x_out11.shape)
        x_perm2 = image[6].permute(0,3,2,1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        # print(x_out2.shape)
        image[6] = x_out2.permute(0,3,2,1).contiguous()
        # print(x_out21.shape)

        x_perm1 = image[7].permute(0,2,1,3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        # print(x_out1.shape)
        image[7] = x_out1.permute(0,2,1,3).contiguous()
        # print(x_out11.shape)
        x_perm2 = image[3].permute(0,3,2,1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        # print(x_out2.shape)
        image[3] = x_out2.permute(0,3,2,1).contiguous()
        # print(x_out21.shape)



        
        image1[0] = 1/4 * (image[1] + image[2] + image[5] + image[6])
        # image1[0] = self.conv2_2(x)
        image1[1] = 1/5 * (image[1] + image[2] + image[3] + image[4] + image[6] + image[7])
        # image1[1] = self.conv2_2(x)
        image1[2] = 1/4 * (image[3] + image[4] + image[5] + image[7])
        # image1[2] = self.conv2_2(x)
        image1[3] = image[5]
        # image1[3] = self.conv2_2(x)
        # image1[0] = self.tri(image1[0])
        # image1[1] = self.tri(image1[1])
        # image1[2] = self.tri(image1[2])
        # image1[3] = self.tri(image1[3])
        

        out = self.iBN(image1, nn.BatchNorm2d(self.out_channels))
        out = [self.silu(x.clone()) for x in out]
        # print(out[0].shape)
        # print(out[1].shape)
        out = self.cat([out[0],out[1],out[2],out[3]], 1)
        # print(out.shape)
        # out = self.tri(out)
        out = self.conv1_3(out)
        # print(out.shape)
        out = self.bn1(out)
        # print(out.shape)
        out = self.silu(out)
        # print(out.shape)
        return out
        # x_perm1 = x.permute(0,2,1,3).contiguous()
        # x_out1 = self.ChannelGateH(x_perm1)
        # x_out11 = x_out1.permute(0,2,1,3).contiguous()
        # x_perm2 = x.permute(0,3,2,1).contiguous()
        # x_out2 = self.ChannelGateW(x_perm2)
        # x_out21 = x_out2.permute(0,3,2,1).contiguous()
        # if not self.no_spatial:
        #     x_out = self.SpatialGate(x)
        #     x_out = (1/3)*(x_out + x_out11 + x_out21)
        # else:
        #     x_out = (1/2)*(x_out11 + x_out21)
        # return x_out










class SEPC_DSC122_2(nn.Module):
    def __init__(self, in_channels, out_channels, e = 0.5, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(SEPC_DSC122_2, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
        self.out_channels = out_channels
        self.in_channels = in_channels
        c_ = int(out_channels * e)
        self.conv1 = nn.Conv2d(in_channels, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False) #降维 + 残差
        self.conv1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 残差
        self.conv1_3 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 降维

        self.conv2_0 = nn.Conv2d(c_, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
        self.conv2_1 = nn.Conv2d(c_, c_, kernel_size=3, padding=1, stride=1, groups=1, bias=False)
        self.conv2_2 = DSConv(c_, c_, kernel_size=3, extend_scope=1, morph=1, if_offset=True)
        self.conv2_3 = AKConv(c_, c_, 3, stride=1, bias=None)
        self.tri = TripletAttention()
        self.cat = torch.cat
        self.silu = nn.SiLU()
        self.interpolate = F.interpolate
        self.split = torch.split
        self.bn = nn.BatchNorm2d(128)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def iBN(self, fms, bn):
        sizes = [p.shape[2:] for p in fms]
        n, c = fms[0].shape[0], fms[0].shape[1]
        fm = torch.cat([p.view(n, c, 1, -1) for p in fms], dim=-1)
        fm = self.bn(fm)
        fm = self.split(fm, [s[0] * s[1] for s in sizes], dim=-1)
        return [p.view(n, c, s[0], s[1]) for p, s in zip(fm, sizes)]
    
    def forward(self, x):
        image = [None] * 6
        image1 = [None] * 4
        x = self.conv1(x)
        # print(x.shape)
        image[0] = self.conv2_0(x)
        # print(image[0].shape)
        image[1] = self.conv2_1(x)
        # print(image[1].shape)
        image[2] = self.conv2_1(self.conv2_1(x))
        # print(image[2].shape)
        image[3] = self.conv2_2(x)
        # print(image[2].shape)
        image[4] = self.conv2_2(self.conv2_2(x))
        # print(image[3].shape)
        image[5] = x
        # print(image[4].shape)
        image1[0] = image[0] + image[1] + image[2] + image[5] 
        # image1[0] = self.conv2_2(x)
        image1[1] = image[1] + image[2] + image[3] + image[4]
        # image1[1] = self.conv2_2(x)
        image1[2] = image[0] + image[3] + image[4] + image[5]
        # image1[2] = self.conv2_2(x)
        image1[3] = image[5]
        # image1[3] = self.conv2_2(x)

        # image1[1] = self.SpatialGate(image1[1])
        # x_perm1 = image1[2].permute(0,2,1,3).contiguous()
        # x_out1 = self.ChannelGateH(x_perm1)
        # # print(x_out1.shape)
        # image1[2] = x_out1.permute(0,2,1,3).contiguous()
        # # print(x_out11.shape)
        # x_perm2 = image1[0].permute(0,3,2,1).contiguous()
        # x_out2 = self.ChannelGateW(x_perm2)
        # # print(x_out2.shape)
        # image1[0] = x_out2.permute(0,3,2,1).contiguous()
        # # print(x_out21.shape)
        # image1[3] = image[5]
        
        out = self.iBN(image1, nn.BatchNorm2d(self.out_channels))
        out = [self.silu(x.clone()) for x in out]
        # print(out[0].shape)
        # print(out[1].shape)
        out = self.cat([out[0],out[1],out[2],out[3]], 1)
        # print(out.shape)
        # out = self.tri(out)
        out = self.conv1_3(out)
        # print(out.shape)
        out = self.bn1(out)
        # print(out.shape)
        out = self.silu(out)
        # print(out.shape)
        return out
        # x_perm1 = x.permute(0,2,1,3).contiguous()
        # x_out1 = self.ChannelGateH(x_perm1)
        # x_out11 = x_out1.permute(0,2,1,3).contiguous()
        # x_perm2 = x.permute(0,3,2,1).contiguous()
        # x_out2 = self.ChannelGateW(x_perm2)
        # x_out21 = x_out2.permute(0,3,2,1).contiguous()
        # if not self.no_spatial:
        #     x_out = self.SpatialGate(x)
        #     x_out = (1/3)*(x_out + x_out11 + x_out21)
        # else:
        #     x_out = (1/2)*(x_out11 + x_out21)
        # return x_out






class SEPC_DSC122_3(nn.Module):
    def __init__(self, in_channels, out_channels, e = 0.5, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(SEPC_DSC122_3, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
        self.out_channels = out_channels
        self.in_channels = in_channels
        c_ = int(out_channels * e)
        self.conv1 = nn.Conv2d(in_channels, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False) #降维 + 残差
        self.conv1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 残差
        self.conv1_3 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 降维

        self.conv2_0 = nn.Conv2d(c_, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
        self.conv2_1 = nn.Conv2d(c_, c_, kernel_size=3, padding=1, stride=1, groups=1, bias=False)
        self.conv2_2 = DSConv(c_, c_, kernel_size=3, extend_scope=1, morph=1, if_offset=True)
        self.conv2_3 = AKConv(c_, c_, 3, stride=1, bias=None)
        self.tri = TripletAttention()
        self.cat = torch.cat
        self.silu = nn.SiLU()
        self.interpolate = F.interpolate
        self.split = torch.split
        self.bn = nn.BatchNorm2d(128)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def iBN(self, fms, bn):
        sizes = [p.shape[2:] for p in fms]
        n, c = fms[0].shape[0], fms[0].shape[1]
        fm = torch.cat([p.view(n, c, 1, -1) for p in fms], dim=-1)
        fm = self.bn(fm)
        fm = self.split(fm, [s[0] * s[1] for s in sizes], dim=-1)
        return [p.view(n, c, s[0], s[1]) for p, s in zip(fm, sizes)]
    
    def forward(self, x):
        image = [None] * 6
        image1 = [None] * 4
        x = self.conv1(x)
        # print(x.shape)
        image[0] = self.conv2_0(x)
        # print(image[0].shape)
        image[1] = self.conv2_1(x)
        # print(image[1].shape)
        image[2] = self.conv2_1(self.conv2_1(x))
        # print(image[2].shape)
        image[3] = self.conv2_2(x)
        # print(image[2].shape)
        image[4] = self.conv2_2(self.conv2_2(x))
        # print(image[3].shape)
        image[5] = x
        # print(image[4].shape)

        image[2] = self.SpatialGate(image[2])
        image[4] = self.SpatialGate(image[4])
        x_perm1 = image[1].permute(0,2,1,3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        # print(x_out1.shape)
        image[1] = x_out1.permute(0,2,1,3).contiguous()
        # print(x_out11.shape)
        x_perm2 = image[3].permute(0,3,2,1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        # print(x_out2.shape)
        image[3] = x_out2.permute(0,3,2,1).contiguous()
        # print(x_out21.shape)

        image1[0] = image[0] + image[1] + image[2] + image[5] 
        # image1[0] = self.conv2_2(x)
        image1[1] = image[1] + image[2] + image[3] + image[4]
        # image1[1] = self.conv2_2(x)
        image1[2] = image[0] + image[3] + image[4] + image[5]
        # image1[2] = self.conv2_2(x)
        image1[3] = image[5]
        # image1[3] = self.conv2_2(x)
        out = self.iBN(image1, nn.BatchNorm2d(self.out_channels))
        out = [self.silu(x.clone()) for x in out]
        # print(out[0].shape)
        # print(out[1].shape)
        out = self.cat([out[0],out[1],out[2],out[3]], 1)
        # print(out.shape)
        # out = self.tri(out)
        out = self.conv1_3(out)
        # print(out.shape)
        out = self.bn1(out)
        # print(out.shape)
        out = self.silu(out)
        # print(out.shape)
        return out
        # x_perm1 = x.permute(0,2,1,3).contiguous()
        # x_out1 = self.ChannelGateH(x_perm1)
        # x_out11 = x_out1.permute(0,2,1,3).contiguous()
        # x_perm2 = x.permute(0,3,2,1).contiguous()
        # x_out2 = self.ChannelGateW(x_perm2)
        # x_out21 = x_out2.permute(0,3,2,1).contiguous()
        # if not self.no_spatial:
        #     x_out = self.SpatialGate(x)
        #     x_out = (1/3)*(x_out + x_out11 + x_out21)
        # else:
        #     x_out = (1/2)*(x_out11 + x_out21)
        # return x_out
    



class SEPC_DSC122_4(nn.Module):
    def __init__(self, in_channels, out_channels, e = 0.5, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(SEPC_DSC122_4, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
        self.out_channels = out_channels
        self.in_channels = in_channels
        c_ = int(out_channels * e)
        self.conv1 = nn.Conv2d(in_channels, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False) #降维 + 残差
        self.conv1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 残差
        self.conv1_3 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 降维

        self.conv2_0 = nn.Conv2d(c_, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
        self.conv2_1 = nn.Conv2d(c_, c_, kernel_size=3, padding=1, stride=1, groups=1, bias=False)
        self.conv2_2 = DSConv(c_, c_, kernel_size=3, extend_scope=1, morph=1, if_offset=True)
        self.conv2_3 = AKConv(c_, c_, 3, stride=1, bias=None)
        self.tri = TripletAttention()
        self.cat = torch.cat
        self.silu = nn.SiLU()
        self.interpolate = F.interpolate
        self.split = torch.split
        self.bn = nn.BatchNorm2d(128)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def iBN(self, fms, bn):
        sizes = [p.shape[2:] for p in fms]
        n, c = fms[0].shape[0], fms[0].shape[1]
        fm = torch.cat([p.view(n, c, 1, -1) for p in fms], dim=-1)
        fm = self.bn(fm)
        fm = self.split(fm, [s[0] * s[1] for s in sizes], dim=-1)
        return [p.view(n, c, s[0], s[1]) for p, s in zip(fm, sizes)]
    
    def forward(self, x):
        image = [None] * 8
        image1 = [None] * 4
        x = self.conv1(x)
        # print(x.shape)
        image[0] = self.conv2_0(x)
        # print(image[0].shape)
        image[1] = self.conv2_1(x)
        # print(image[1].shape)
        image[2] = self.conv2_1(self.conv2_1(x))
        # print(image[2].shape)
        image[3] = self.conv2_2(x)
        # print(image[2].shape)
        image[4] = self.conv2_2(self.conv2_2(x))
        # print(image[3].shape)
        image[5] = x
        image[6] = image[1]
        image[7] = image[3]
        # print(image[4].shape)
        
        image[2] = self.SpatialGate(image[2])
        image[4] = self.SpatialGate(image[4])

        x_perm1 = image[1].permute(0,2,1,3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        # print(x_out1.shape)
        image[1] = x_out1.permute(0,2,1,3).contiguous()
        # print(x_out11.shape)
        x_perm2 = image[6].permute(0,3,2,1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        # print(x_out2.shape)
        image[6] = x_out2.permute(0,3,2,1).contiguous()
        # print(x_out21.shape)

        x_perm1 = image[7].permute(0,2,1,3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        # print(x_out1.shape)
        image[7] = x_out1.permute(0,2,1,3).contiguous()
        # print(x_out11.shape)
        x_perm2 = image[3].permute(0,3,2,1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        # print(x_out2.shape)
        image[3] = x_out2.permute(0,3,2,1).contiguous()
        # print(x_out21.shape)



        
        image1[0] = image[0] + image[1] + image[2] + image[5] + image[6]
        # image1[0] = self.conv2_2(x)
        image1[1] = image[1] + image[2] + image[3] + image[4] + image[6] + image[7]
        # image1[1] = self.conv2_2(x)
        image1[2] = image[0] + image[3] + image[4] + image[5] + image[7]
        # image1[2] = self.conv2_2(x)
        image1[3] = image[5]
        # image1[3] = self.conv2_2(x)
        # image1[0] = self.tri(image1[0])
        # image1[1] = self.tri(image1[1])
        # image1[2] = self.tri(image1[2])
        # image1[3] = self.tri(image1[3])

        out = self.iBN(image1, nn.BatchNorm2d(self.out_channels))
        out = [self.silu(x.clone()) for x in out]
        # print(out[0].shape)
        # print(out[1].shape)
        out = self.cat([out[0],out[1],out[2],out[3]], 1)
        # print(out.shape)
        # out = self.tri(out)
        out = self.conv1_3(out)
        # print(out.shape)
        out = self.bn1(out)
        # print(out.shape)
        out = self.silu(out)
        # print(out.shape)
        return out
        # x_perm1 = x.permute(0,2,1,3).contiguous()
        # x_out1 = self.ChannelGateH(x_perm1)
        # x_out11 = x_out1.permute(0,2,1,3).contiguous()
        # x_perm2 = x.permute(0,3,2,1).contiguous()
        # x_out2 = self.ChannelGateW(x_perm2)
        # x_out21 = x_out2.permute(0,3,2,1).contiguous()
        # if not self.no_spatial:
        #     x_out = self.SpatialGate(x)
        #     x_out = (1/3)*(x_out + x_out11 + x_out21)
        # else:
        #     x_out = (1/2)*(x_out11 + x_out21)
        # return x_out
    




class SEPC_DSC122_5(nn.Module):
    def __init__(self, in_channels, out_channels, e = 0.5, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(SEPC_DSC122_5, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
        self.out_channels = out_channels
        self.in_channels = in_channels
        c_ = int(out_channels * e)
        self.conv1 = nn.Conv2d(in_channels, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False) #降维 + 残差
        self.conv1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 残差
        self.conv1_3 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 降维

        self.conv2_0 = nn.Conv2d(c_, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
        self.conv2_1 = nn.Conv2d(c_, c_, kernel_size=3, padding=1, stride=1, groups=1, bias=False)
        self.conv2_2 = DSConv(c_, c_, kernel_size=3, extend_scope=1, morph=1, if_offset=True)
        self.conv2_3 = AKConv(c_, c_, 3, stride=1, bias=None)
        self.tri = TripletAttention()
        self.cat = torch.cat
        self.silu = nn.SiLU()
        self.interpolate = F.interpolate
        self.split = torch.split
        self.bn = nn.BatchNorm2d(128)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def iBN(self, fms, bn):
        sizes = [p.shape[2:] for p in fms]
        n, c = fms[0].shape[0], fms[0].shape[1]
        fm = torch.cat([p.view(n, c, 1, -1) for p in fms], dim=-1)
        fm = self.bn(fm)
        fm = self.split(fm, [s[0] * s[1] for s in sizes], dim=-1)
        return [p.view(n, c, s[0], s[1]) for p, s in zip(fm, sizes)]
    
    def forward(self, x):
        image = [None] * 8
        image1 = [None] * 4
        x = self.conv1(x)
        # print(x.shape)
        image[0] = self.conv2_0(x)
        # print(image[0].shape)
        image[1] = self.conv2_1(x)
        # print(image[1].shape)
        image[2] = self.conv2_1(self.conv2_1(x))
        # print(image[2].shape)
        image[3] = self.conv2_2(x)
        # print(image[2].shape)
        image[4] = self.conv2_2(self.conv2_2(x))
        # print(image[3].shape)
        image[5] = x
        image[6] = image[1]
        image[7] = image[3]
        # print(image[4].shape)
        
        image[2] = self.SpatialGate(image[2])
        image[4] = self.SpatialGate(image[4])

        x_perm1 = image[1].permute(0,2,1,3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        # print(x_out1.shape)
        image[1] = x_out1.permute(0,2,1,3).contiguous()
        # print(x_out11.shape)
        x_perm2 = image[6].permute(0,3,2,1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        # print(x_out2.shape)
        image[6] = x_out2.permute(0,3,2,1).contiguous()
        # print(x_out21.shape)

        x_perm1 = image[7].permute(0,2,1,3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        # print(x_out1.shape)
        image[7] = x_out1.permute(0,2,1,3).contiguous()
        # print(x_out11.shape)
        x_perm2 = image[3].permute(0,3,2,1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        # print(x_out2.shape)
        image[3] = x_out2.permute(0,3,2,1).contiguous()
        # print(x_out21.shape)



        
        image1[0] = 1/4 *  (image[1] + image[2] + image[5] + image[6])
        # image1[0] = self.conv2_2(x)
        image1[1] = 1/5 * (image[1] + image[2] + image[3] + image[4] + image[6] + image[7])
        # image1[1] = self.conv2_2(x)
        image1[2] = 1/4 * (image[3] + image[4] + image[5] + image[7])
        # image1[2] = self.conv2_2(x)
        image1[3] = image[5]
        # image1[3] = self.conv2_2(x)
        # image1[0] = self.tri(image1[0])
        # image1[1] = self.tri(image1[1])
        # image1[2] = self.tri(image1[2])
        # image1[3] = self.tri(image1[3])

        out = self.iBN(image1, nn.BatchNorm2d(self.out_channels))
        out = [self.silu(x.clone()) for x in out]
        # print(out[0].shape)
        # print(out[1].shape)
        out = self.cat([out[0],out[1],out[2],out[3]], 1)
        # print(out.shape)
        # out = self.tri(out)
        out = self.conv1_3(out)
        # print(out.shape)
        out = self.bn1(out)
        # print(out.shape)
        out = self.silu(out)
        # print(out.shape)
        return out
        # x_perm1 = x.permute(0,2,1,3).contiguous()
        # x_out1 = self.ChannelGateH(x_perm1)
        # x_out11 = x_out1.permute(0,2,1,3).contiguous()
        # x_perm2 = x.permute(0,3,2,1).contiguous()
        # x_out2 = self.ChannelGateW(x_perm2)
        # x_out21 = x_out2.permute(0,3,2,1).contiguous()
        # if not self.no_spatial:
        #     x_out = self.SpatialGate(x)
        #     x_out = (1/3)*(x_out + x_out11 + x_out21)
        # else:
        #     x_out = (1/2)*(x_out11 + x_out21)
        # return x_out





from models.akconv import AKConv
class SEPC_AK122(nn.Module):
    def __init__(self, in_channels, out_channels, e = 0.5, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(SEPC_AK122, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
        self.out_channels = out_channels
        self.in_channels = in_channels
        c_ = int(out_channels * e)
        self.conv1 = nn.Conv2d(in_channels, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False) #降维 + 残差
        self.conv1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 残差
        self.conv1_3 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 降维

        self.conv2_0 = nn.Conv2d(c_, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
        self.conv2_1 = nn.Conv2d(c_, c_, kernel_size=3, padding=1, stride=1, groups=1, bias=False)
        self.conv2_2 = AKConv(c_, c_, 3, stride=1, bias=None)
        self.conv2_3 = AKConv(c_, c_, 3, stride=1, bias=None)
        self.tri = TripletAttention()
        self.cat = torch.cat
        self.silu = nn.SiLU()
        self.interpolate = F.interpolate
        self.split = torch.split
        self.bn = nn.BatchNorm2d(128)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def iBN(self, fms, bn):
        sizes = [p.shape[2:] for p in fms]
        n, c = fms[0].shape[0], fms[0].shape[1]
        fm = torch.cat([p.view(n, c, 1, -1) for p in fms], dim=-1)
        fm = self.bn(fm)
        fm = self.split(fm, [s[0] * s[1] for s in sizes], dim=-1)
        return [p.view(n, c, s[0], s[1]) for p, s in zip(fm, sizes)]
    
    def forward(self, x):
        image = [None] * 6
        image1 = [None] * 4
        x = self.conv1(x)
        # print(x.shape)
        image[0] = self.conv2_0(x)
        # print(image[0].shape)
        image[1] = self.conv2_1(x)
        # print(image[1].shape)
        image[2] = self.conv2_1(self.conv2_1(x))
        # print(image[2].shape)
        image[3] = self.conv2_2(x)
        # print(image[2].shape)
        image[4] = self.conv2_2(self.conv2_2(x))
        # print(image[3].shape)
        image[5] = x
        # print(image[4].shape)
        image1[0] = image[0] + image[1] + image[2] + image[5] 
        # image1[0] = self.conv2_2(x)
        image1[1] = image[1] + image[2] + image[3] + image[4]
        # image1[1] = self.conv2_2(x)
        image1[2] = image[0] + image[3] + image[4] + image[5]
        # image1[2] = self.conv2_2(x)
        image1[3] = image[5]
        # image1[3] = self.conv2_2(x)
        # image1[0] = self.tri(image1[0])
        # image1[1] = self.tri(image1[1])
        # image1[2] = self.tri(image1[2])
        # image1[3] = self.tri(image1[3])
        
        image1[1] = self.SpatialGate(image1[1])
        x_perm1 = image1[2].permute(0,2,1,3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        # print(x_out1.shape)
        image1[2] = x_out1.permute(0,2,1,3).contiguous()
        # print(x_out11.shape)
        x_perm2 = image1[0].permute(0,3,2,1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        # print(x_out2.shape)
        image1[0] = x_out2.permute(0,3,2,1).contiguous()
        # print(x_out21.shape)
        image1[3] = image[5]

        out = self.iBN(image1, nn.BatchNorm2d(self.out_channels))
        out = [self.silu(x.clone()) for x in out]
        # print(out[0].shape)
        # print(out[1].shape)
        out = self.cat([out[0],out[1],out[2],out[3]], 1)
        # print(out.shape)
        # out = self.tri(out)
        out = self.conv1_3(out)
        # print(out.shape)
        out = self.bn1(out)
        # print(out.shape)
        out = self.silu(out)
        # print(out.shape)
        return out
        # x_perm1 = x.permute(0,2,1,3).contiguous()
        # x_out1 = self.ChannelGateH(x_perm1)
        # x_out11 = x_out1.permute(0,2,1,3).contiguous()
        # x_perm2 = x.permute(0,3,2,1).contiguous()
        # x_out2 = self.ChannelGateW(x_perm2)
        # x_out21 = x_out2.permute(0,3,2,1).contiguous()
        # if not self.no_spatial:
        #     x_out = self.SpatialGate(x)
        #     x_out = (1/3)*(x_out + x_out11 + x_out21)
        # else:
        #     x_out = (1/2)*(x_out11 + x_out21)
        # return x_out












from models.akconv import AKConv
class SEPC_AK_new(nn.Module):
    def __init__(self, in_channels, out_channels, e = 0.5, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(SEPC_AK_new, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
        self.out_channels = out_channels
        self.in_channels = in_channels
        c_ = int(out_channels * e)
        self.conv1 = nn.Conv2d(in_channels, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False) #降维 + 残差
        self.conv1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 残差
        self.conv1_3 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 降维

        self.conv2_0 = nn.Conv2d(c_, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
        self.conv2_1 = nn.Conv2d(c_, c_, kernel_size=3, padding=1, stride=1, groups=1, bias=False)
        self.conv2_2 = AKConv(c_, c_, 3, stride=1, bias=None)
        self.conv2_3 = AKConv(c_, c_, 3, stride=1, bias=None)
        self.tri = TripletAttention()
        self.cat = torch.cat
        self.silu = nn.SiLU()
        self.interpolate = F.interpolate
        self.split = torch.split
        self.bn = nn.BatchNorm2d(128)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def iBN(self, fms, bn):
        sizes = [p.shape[2:] for p in fms]
        n, c = fms[0].shape[0], fms[0].shape[1]
        fm = torch.cat([p.view(n, c, 1, -1) for p in fms], dim=-1)
        fm = self.bn(fm)
        fm = self.split(fm, [s[0] * s[1] for s in sizes], dim=-1)
        return [p.view(n, c, s[0], s[1]) for p, s in zip(fm, sizes)]
    
    def forward(self, x):
        image = [None] * 6
        image1 = [None] * 4
        x = self.conv1(x)
        # print(x.shape)
        image[0] = self.conv2_0(x)
        # print(image[0].shape)
        image[1] = self.conv2_1(x)
        # print(image[1].shape)
        image[2] = self.conv2_1(self.conv2_1(x))
        # print(image[2].shape)
        image[3] = self.conv2_2(x)
        # print(image[2].shape)
        image[4] = self.conv2_2(self.conv2_2(x))
        # print(image[3].shape)
        image[5] = x
        # print(image[4].shape)
        image1[0] = image[0] + image[1] + image[2] + image[5] 
        # image1[0] = self.conv2_2(x)
        image1[1] = image[1] + image[2] + image[3] + image[4]
        # image1[1] = self.conv2_2(x)
        image1[2] = image[0] + image[3] + image[4] + image[5]
        # image1[2] = self.conv2_2(x)
        image1[3] = image[5]
        # image1[3] = self.conv2_2(x)
        image1[0] = self.tri(image1[0])
        image1[1] = self.tri(image1[1])
        image1[2] = self.tri(image1[2])
        image1[3] = self.tri(image1[3])
        out = self.iBN(image1, nn.BatchNorm2d(self.out_channels))
        out = [self.silu(x.clone()) for x in out]
        # print(out[0].shape)
        # print(out[1].shape)
        out = self.cat([out[0],out[1],out[2],out[3]], 1)
        # print(out.shape)
        # out = self.tri(out)
        out = self.conv1_3(out)
        # print(out.shape)
        out = self.bn1(out)
        # print(out.shape)
        out = self.silu(out)
        # print(out.shape)
        return out
        # x_perm1 = x.permute(0,2,1,3).contiguous()
        # x_out1 = self.ChannelGateH(x_perm1)
        # x_out11 = x_out1.permute(0,2,1,3).contiguous()
        # x_perm2 = x.permute(0,3,2,1).contiguous()
        # x_out2 = self.ChannelGateW(x_perm2)
        # x_out21 = x_out2.permute(0,3,2,1).contiguous()
        # if not self.no_spatial:
        #     x_out = self.SpatialGate(x)
        #     x_out = (1/3)*(x_out + x_out11 + x_out21)
        # else:
        #     x_out = (1/2)*(x_out11 + x_out21)
        # return x_out










from models.akconv import AKConv
class SEPC_T2_s(nn.Module):
    def __init__(self, in_channels, out_channels, e = 0.5, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(SEPC_T2_s, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
        self.out_channels = out_channels
        self.in_channels = in_channels
        c_ = int(out_channels * e)
        self.conv1 = nn.Conv2d(in_channels, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False) #降维 + 残差
        self.conv1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 残差
        self.conv1_3 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 降维

        self.conv2_0 = nn.Conv2d(c_, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
        self.conv2_1 = nn.Conv2d(c_, c_, kernel_size=3, padding=1, stride=1, groups=1, bias=False)
        self.conv2_2 = AKConv(c_, c_, 3, stride=1, bias=None)
        self.conv2_3 = AKConv(c_, c_, 5, stride=1, bias=None)
        self.cat = torch.cat
        self.silu = nn.SiLU()
        self.interpolate = F.interpolate
        self.split = torch.split
        self.bn = nn.BatchNorm2d(128)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def iBN(self, fms, bn):
        sizes = [p.shape[2:] for p in fms]
        n, c = fms[0].shape[0], fms[0].shape[1]
        fm = torch.cat([p.view(n, c, 1, -1) for p in fms], dim=-1)
        fm = self.bn(fm)
        fm = self.split(fm, [s[0] * s[1] for s in sizes], dim=-1)
        return [p.view(n, c, s[0], s[1]) for p, s in zip(fm, sizes)]
    
    def forward(self, x):
        image = [None] * 5
        image1 = [None] * 4
        x = self.conv1(x)
        # print(x.shape) (1,128,32,32)
        image[0] = self.conv2_0(x)
        # print(image[0].shape) (1,128,32,32)
        image[1] = self.conv2_1(x)
        # print(image[1].shape)
        image[2] = self.conv2_2(x)
        # print(image[2].shape)
        image[3] = self.conv2_3(x)
        # print(image[3].shape)
        image[4] = x
        print(image[4].shape)
        x_out = self.SpatialGate(image[3])
        x_perm1 = image[1].permute(0,2,1,3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        # print(x_out1.shape)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        # print(x_out11.shape)
        x_perm2 = image[2].permute(0,3,2,1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        # print(x_out2.shape)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()
        # print(x_out21.shape)
        image1[0] = x + x_out + x_out21
        image1[0] = self.conv2_2(x)
        image1[1] = image[0] + x_out11 + x_out21 + x_out
        image1[1] = self.conv2_2(x)
        image1[2] = x + x_out11 + image[0]
        image1[2] = self.conv2_2(x)
        image1[3] = x
        image1[3] = self.conv2_2(x)
        out = self.iBN(image1, nn.BatchNorm2d(self.out_channels))
        out = [self.silu(x.clone()) for x in out]
        # print(out[0].shape)
        # print(out[1].shape)
        out = self.cat([out[0],out[1],out[2],out[3]], 1)
        print(out.shape)
        out = self.conv1_3(out)
        print(out.shape)
        out = self.bn1(out)
        # print(out.shape)
        out = self.silu(out)
        # print(out.shape)
        return out
        # x_perm1 = x.permute(0,2,1,3).contiguous()
        # x_out1 = self.ChannelGateH(x_perm1)
        # x_out11 = x_out1.permute(0,2,1,3).contiguous()
        # x_perm2 = x.permute(0,3,2,1).contiguous()
        # x_out2 = self.ChannelGateW(x_perm2)
        # x_out21 = x_out2.permute(0,3,2,1).contiguous()
        # if not self.no_spatial:
        #     x_out = self.SpatialGate(x)
        #     x_out = (1/3)*(x_out + x_out11 + x_out21)
        # else:
        #     x_out = (1/2)*(x_out11 + x_out21)
        # return x_out




# ------------------------------------CARAFE -----start--------------------------------
class CARAFE(nn.Module):
    #CARAFE: Content-Aware ReAssembly of FEatures       https://arxiv.org/pdf/1905.02188.pdf
    def __init__(self, c1, c2, kernel_size=3, up_factor=2):
        super(CARAFE, self).__init__()
        self.kernel_size = kernel_size
        self.up_factor = up_factor
        self.down = nn.Conv2d(c1, c1 // 4, 1)
        self.encoder = nn.Conv2d(c1 // 4, self.up_factor ** 2 * self.kernel_size ** 2,
                                 self.kernel_size, 1, self.kernel_size // 2)
        self.out = nn.Conv2d(c1, c2, 1)
 
    def forward(self, x):
        N, C, H, W = x.size()
        # N,C,H,W -> N,C,delta*H,delta*W
        # kernel prediction module
        kernel_tensor = self.down(x)  # (N, Cm, H, W)
        kernel_tensor = self.encoder(kernel_tensor)  # (N, S^2 * Kup^2, H, W)
        kernel_tensor = F.pixel_shuffle(kernel_tensor, self.up_factor)  # (N, S^2 * Kup^2, H, W)->(N, Kup^2, S*H, S*W)
        kernel_tensor = F.softmax(kernel_tensor, dim=1)  # (N, Kup^2, S*H, S*W)
        kernel_tensor = kernel_tensor.unfold(2, self.up_factor, step=self.up_factor) # (N, Kup^2, H, W*S, S)
        kernel_tensor = kernel_tensor.unfold(3, self.up_factor, step=self.up_factor) # (N, Kup^2, H, W, S, S)
        kernel_tensor = kernel_tensor.reshape(N, self.kernel_size ** 2, H, W, self.up_factor ** 2) # (N, Kup^2, H, W, S^2)
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, Kup^2, S^2)
 
        # content-aware reassembly module
        # tensor.unfold: dim, size, step
        x = F.pad(x, pad=(self.kernel_size // 2, self.kernel_size // 2,
                                          self.kernel_size // 2, self.kernel_size // 2),
                          mode='constant', value=0) # (N, C, H+Kup//2+Kup//2, W+Kup//2+Kup//2)
        x = x.unfold(2, self.kernel_size, step=1) # (N, C, H, W+Kup//2+Kup//2, Kup)
        x = x.unfold(3, self.kernel_size, step=1) # (N, C, H, W, Kup, Kup)
        x = x.reshape(N, C, H, W, -1) # (N, C, H, W, Kup^2)
        x = x.permute(0, 2, 3, 1, 4)  # (N, H, W, C, Kup^2)
 
        out_tensor = torch.matmul(x, kernel_tensor)  # (N, H, W, C, S^2)
        out_tensor = out_tensor.reshape(N, H, W, -1)
        out_tensor = out_tensor.permute(0, 3, 1, 2)
        out_tensor = F.pixel_shuffle(out_tensor, self.up_factor)
        out_tensor = self.out(out_tensor)
        #print("up shape:",out_tensor.shape)
        return out_tensor
# ------------------------------------CARAFE -----start--------------------------------



class ODConv2d_3rd(nn.Conv2d):
 
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 K=4, r=1 / 16, save_parameters=False,
                 padding_mode='zeros', device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.K = K
        self.r = r
        self.save_parameters = save_parameters
 
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode)
 
        del self.weight
        self.weight = nn.Parameter(torch.empty((
            K,
            out_channels,
            in_channels // groups,
            *self.kernel_size,
        ), **factory_kwargs))
 
        if bias:
            del self.bias
            self.bias = nn.Parameter(torch.empty(K, out_channels, **factory_kwargs))
 
        hidden_dim = max(int(in_channels * r), 16)  #设置下限为16
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.reduction = nn.Linear(in_channels, hidden_dim)
        self.fc = nn.Conv2d(in_channels, hidden_dim, 1, bias = False)
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.act = nn.ReLU(inplace=True)
        # self.act = nn.SiLU(inplace=True)
 
        self.fc_f = nn.Linear(hidden_dim, out_channels)
        if not save_parameters or self.kernel_size[0] * self.kernel_size[1] > 1:
            self.fc_s = nn.Linear(hidden_dim, self.kernel_size[0] * self.kernel_size[1])
        if not save_parameters or in_channels // groups > 1:
            self.fc_c = nn.Linear(hidden_dim, in_channels // groups)
        if not save_parameters or K > 1:
            self.fc_w = nn.Linear(hidden_dim, K)
 
        self.reset_parameters()
 
    def reset_parameters(self) -> None:
        fan_out = self.kernel_size[0] * self.kernel_size[1] * self.out_channels // self.groups
        for i in range(self.K):
            self.weight.data[i].normal_(0, math.sqrt(2.0 / fan_out))
        if self.bias is not None:
            self.bias.data.zero_()
 
    def extra_repr(self):
        return super().extra_repr() + f', K={self.K}, r={self.r:.4}'
 
    def get_weight_bias(self, context):
        B, C, H, W = context.shape
 
        if C != self.in_channels:
            raise ValueError(
                f"Expected context{[B, C, H, W]} to have {self.in_channels} channels, but got {C} channels instead")
 
        # x = self.gap(context).squeeze(-1).squeeze(-1)  # B, c_in
        # x = self.reduction(x)  # B, hidden_dim
        x = self.gap(context)
        x = self.fc(x)
        if x.size(0)>1:
            x = self.bn(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.act(x)
 
        attn_f = self.fc_f(x).sigmoid()  # B, c_out
        attn = attn_f.view(B, 1, -1, 1, 1, 1)  # B, 1, c_out, 1, 1, 1
        if hasattr(self, 'fc_s'):
            attn_s = self.fc_s(x).sigmoid()  # B, k * k
            attn = attn * attn_s.view(B, 1, 1, 1, *self.kernel_size)  # B, 1, c_out, 1, k, k
        if hasattr(self, 'fc_c'):
            attn_c = self.fc_c(x).sigmoid()  # B, c_in // groups
            attn = attn * attn_c.view(B, 1, 1, -1, 1, 1)  # B, 1, c_out, c_in // groups, k, k
        if hasattr(self, 'fc_w'):
            attn_w = self.fc_w(x).softmax(-1)  # B, n
            attn = attn * attn_w.view(B, -1, 1, 1, 1, 1)  # B, n, c_out, c_in // groups, k, k
 
        weight = (attn * self.weight).sum(1)  # B, c_out, c_in // groups, k, k
        weight = weight.view(-1, self.in_channels // self.groups, *self.kernel_size)  # B * c_out, c_in // groups, k, k
 
        bias = None
        if self.bias is not None:
            if hasattr(self, 'fc_w'):
                bias = attn_w @ self.bias
            else:
                bias = self.bias.tile(B, 1)
            bias = bias.view(-1)  # B * c_out
 
        return weight, bias
 
    def forward(self, input, context=None):
        B, C, H, W = input.shape
 
        if C != self.in_channels:
            raise ValueError(
                f"Expected input{[B, C, H, W]} to have {self.in_channels} channels, but got {C} channels instead")
 
        weight, bias = self.get_weight_bias(context or input)
 
        output = nn.functional.conv2d(
            input.view(1, B * C, H, W), weight, bias,
            self.stride, self.padding, self.dilation, B * self.groups)  # 1, B * c_out, h_out, w_out
        output = output.view(B, self.out_channels, *output.shape[2:])
 
        return output
 
    def debug(self, input, context=None):
        B, C, H, W = input.shape
 
        if C != self.in_channels:
            raise ValueError(
                f"Expected input{[B, C, H, W]} to have {self.in_channels} channels, but got {C} channels instead")
 
        output_size = [
            ((H, W)[i] + 2 * self.padding[i] - self.dilation[i] * (self.kernel_size[i] - 1) - 1) // self.stride[i] + 1
            for i in range(2)
        ]
 
        weight, bias = self.get_weight_bias(context or input)
 
        weight = weight.view(B, self.groups, self.out_channels // self.groups, -1)  # B, groups, c_out // groups, c_in // groups * k * k
 
        unfold = nn.functional.unfold(
            input, self.kernel_size, self.dilation, self.padding, self.stride)  # B, c_in * k * k, H_out * W_out
        unfold = unfold.view(B, self.groups, -1, output_size[0] * output_size[1])  # B, groups, c_in // groups * k * k, H_out * W_out
 
        output = weight @ unfold  # B, groups, c_out // groups, H_out * W_out
        output = output.view(B, self.out_channels, *output_size)  # B, c_out, H_out * W_out
 
        if bias is not None:
            output = output + bias.view(B, self.out_channels, 1, 1)
 
        return output
 
class ODConv_3rd(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, kerNums=1, g=1, p=None, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = ODConv2d_3rd(c1, c2, k, s, autopad(k, p), groups=g, K=kerNums)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
 
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
 
    def forward_fuse(self, x):
        return self.act(self.conv(x))


 
import torch.nn.functional as F
 
class SCConv(nn.Module):
    def __init__(self, inplanes, planes, stride, padding, dilation, groups, pooling_r, norm_layer):
        super(SCConv, self).__init__()
        self.k2 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r), 
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k3 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k4 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
 
    def forward(self, x):
        identity = x
 
        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:]))) # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out) # k3 * sigmoid(identity + k2)
        out = self.k4(out) # k4
 
        return out





class dw1(nn.Module): 
    def __init__(self, in_channels, out_channels, e = 0.5):
        super().__init__()
        self.out_channels = out_channels
        c_ = int(out_channels * e)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=2, groups=in_channels, bias=False) 


        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
        self.silu = nn.SiLU()
        self.GELU = nn.GELU()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.GELU(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.silu(x)
        return x








# class SEPC_T3(nn.Module):
#     def __init__(self, in_channels, out_channels, e = 0.5, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
#         super(SEPC_T3, self).__init__()
#         self.ChannelGateH = SpatialGate()
#         self.ChannelGateW = SpatialGate()
#         self.no_spatial=no_spatial
#         if not no_spatial:
#             self.SpatialGate = SpatialGate()
#         self.out_channels = out_channels
#         self.in_channels = in_channels
#         c_ = int(out_channels * e)
#         self.conv1 = nn.Conv2d(in_channels, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False) #降维 + 残差
#         self.conv1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
#                                  bias=False)  # 残差
#         self.conv1_3 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
#                                  bias=False)  # 降维

#         self.conv2_0 = nn.Conv2d(c_, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
#         self.conv2_1 = nn.Conv2d(c_, c_, kernel_size=3, padding=1, stride=1, groups=1, bias=False)
#         self.conv2_2 = SCConv(c_, c_, stride=1, padding=1, dilation=1, groups=1, pooling_r=4, norm_layer=nn.BatchNorm2d)

#         self.conv2_3 = SCConv(c_, c_, stride=1, padding=1, dilation=1, groups=1, pooling_r=4, norm_layer=nn.BatchNorm2d)

#         self.cat = torch.cat
#         self.silu = nn.SiLU()
#         self.interpolate = F.interpolate
#         self.split = torch.split
#         self.bn = nn.BatchNorm2d(128)
#         self.bn1 = nn.BatchNorm2d(out_channels)

#     def iBN(self, fms, bn):
#         sizes = [p.shape[2:] for p in fms]
#         n, c = fms[0].shape[0], fms[0].shape[1]
#         fm = torch.cat([p.view(n, c, 1, -1) for p in fms], dim=-1)
#         fm = self.bn(fm)
#         fm = self.split(fm, [s[0] * s[1] for s in sizes], dim=-1)
#         return [p.view(n, c, s[0], s[1]) for p, s in zip(fm, sizes)]
    
#     def forward(self, x):
#         image = [None] * 5
#         image1 = [None] * 4
#         x = self.conv1(x)
#         # print(x.shape)
#         image[0] = self.conv2_0(x)
#         # print(image[0].shape)
#         image[1] = self.conv2_1(x)
#         # print(image[1].shape)
#         image[2] = self.conv2_2(x)
#         # print(image[2].shape)
#         image[3] = self.conv2_3(x)
#         # print(image[3].shape)
#         image[4] = x
#         # print(image[4].shape)
#         x_out = self.SpatialGate(image[3])
#         x_perm1 = image[1].permute(0,2,1,3).contiguous()
#         x_out1 = self.ChannelGateH(x_perm1)
#         # print(x_out1.shape)
#         x_out11 = x_out1.permute(0,2,1,3).contiguous()
#         # print(x_out11.shape)
#         x_perm2 = image[2].permute(0,3,2,1).contiguous()
#         x_out2 = self.ChannelGateW(x_perm2)
#         # print(x_out2.shape)
#         x_out21 = x_out2.permute(0,3,2,1).contiguous()
#         # print(x_out21.shape)
#         image1[0] = x + x_out + x_out21
#         image1[0] = self.conv2_2(x)
#         image1[1] = image[0] + x_out11 + x_out21 + x_out
#         image1[1] = self.conv2_2(x)
#         image1[2] = x + x_out11 + image[0]
#         image1[2] = self.conv2_2(x)
#         image1[3] = x
#         image1[3] = self.conv2_2(x)
#         out = self.iBN(image1, nn.BatchNorm2d(self.out_channels))
#         out = [self.silu(x.clone()) for x in out]
#         # print(out[0].shape)
#         # print(out[1].shape)
#         out = self.cat([out[0],out[1],out[2],out[3]], 1)
#         out = self.conv1_3(out)
#         # print(out.shape)
#         out = self.bn1(out)
#         # print(out.shape)
#         out = self.silu(out)
#         # print(out.shape)
#         return out
#         # x_perm1 = x.permute(0,2,1,3).contiguous()
#         # x_out1 = self.ChannelGateH(x_perm1)
#         # x_out11 = x_out1.permute(0,2,1,3).contiguous()
#         # x_perm2 = x.permute(0,3,2,1).contiguous()
#         # x_out2 = self.ChannelGateW(x_perm2)
#         # x_out21 = x_out2.permute(0,3,2,1).contiguous()
#         # if not self.no_spatial:
#         #     x_out = self.SpatialGate(x)
#         #     x_out = (1/3)*(x_out + x_out11 + x_out21)
#         # else:
#         #     x_out = (1/2)*(x_out11 + x_out21)
#         # return x_out











class SPDConv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, in_channels, out_channels, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()  # Corrected super call
        c1 = in_channels
        c2 = out_channels
        c1 = c1 * 4
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        return self.act(self.conv(x))





import warnings
import torch
from torch import nn

warnings.filterwarnings("ignore")

"""
This code is mainly the deformation process of our DSConv
"""


class DSConv(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, extend_scope, morph,
                 if_offset):
        """
        动态蛇形卷积
        :param in_ch: 输入通道
        :param out_ch: 输出通道
        :param kernel_size: 卷积核的大小
        :param extend_scope: 扩展范围（默认为此方法的1）
        :param morph: 卷积核的形态主要分为两种类型，沿x轴（0）和沿y轴（1）（详细信息请参阅论文）
        :param if_offset: 是否需要变形，如果为False，则是标准卷积核
        """
        super(DSConv, self).__init__()
        # use the <offset_conv> to learn the deformable offset
        self.offset_conv = nn.Conv2d(in_ch, 2 * kernel_size, 3, padding=1)
        self.bn = nn.BatchNorm2d(2 * kernel_size)
        self.kernel_size = kernel_size

        # two types of the DSConv (along x-axis and y-axis)
        self.dsc_conv_x = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(kernel_size, 1),
            stride=(kernel_size, 1),
            padding=0,
        )
        self.dsc_conv_y = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(1, kernel_size),
            stride=(1, kernel_size),
            padding=0,
        )

        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset

    def forward(self, f):
        offset = self.offset_conv(f)
        offset = self.bn(offset)
        # We need a range of deformation between -1 and 1 to mimic the snake's swing
        offset = torch.tanh(offset)
        input_shape = f.shape
        dsc = DSC(input_shape, self.kernel_size, self.extend_scope, self.morph)
        deformed_feature = dsc.deform_conv(f, offset, self.if_offset)
        if self.morph == 0:
            x = self.dsc_conv_x(deformed_feature.type(f.dtype))
            x = self.gn(x)
            x = self.relu(x)
            return x
        else:
            x = self.dsc_conv_y(deformed_feature.type(f.dtype))
            x = self.gn(x)
            x = self.relu(x)
            return x


# Core code, for ease of understanding, we mark the dimensions of input and output next to the code
class DSC(object):

    def __init__(self, input_shape, kernel_size, extend_scope, morph):
        self.num_points = kernel_size
        self.width = input_shape[2]
        self.height = input_shape[3]
        self.morph = morph
        self.extend_scope = extend_scope  # offset (-1 ~ 1) * extend_scope

        # define feature map shape
        """
        B: Batch size  C: Channel  W: Width  H: Height
        """
        self.num_batch = input_shape[0]
        self.num_channels = input_shape[1]

    """
    input: offset [B,2*K,W,H]  K: Kernel size (2*K: 2D image, deformation contains <x_offset> and <y_offset>)
    output_x: [B,1,W,K*H]   coordinate map
    output_y: [B,1,K*W,H]   coordinate map
    """

    def _coordinate_map_3D(self, offset, if_offset):
        device = offset.device
        # offset
        y_offset, x_offset = torch.split(offset, self.num_points, dim=1)

        y_center = torch.arange(0, self.width).repeat([self.height])
        y_center = y_center.reshape(self.height, self.width)
        y_center = y_center.permute(1, 0)
        y_center = y_center.reshape([-1, self.width, self.height])
        y_center = y_center.repeat([self.num_points, 1, 1]).float()
        y_center = y_center.unsqueeze(0)

        x_center = torch.arange(0, self.height).repeat([self.width])
        x_center = x_center.reshape(self.width, self.height)
        x_center = x_center.permute(0, 1)
        x_center = x_center.reshape([-1, self.width, self.height])
        x_center = x_center.repeat([self.num_points, 1, 1]).float()
        x_center = x_center.unsqueeze(0)

        if self.morph == 0:
            """
            Initialize the kernel and flatten the kernel
                y: only need 0
                x: -num_points//2 ~ num_points//2 (Determined by the kernel size)
                !!! The related PPT will be submitted later, and the PPT will contain the whole changes of each step
            """
            y = torch.linspace(0, 0, 1)
            x = torch.linspace(
                -int(self.num_points // 2),
                int(self.num_points // 2),
                int(self.num_points),
            )

            y, x = torch.meshgrid(y, x)
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)

            y_grid = y_spread.repeat([1, self.width * self.height])
            y_grid = y_grid.reshape([self.num_points, self.width, self.height])
            y_grid = y_grid.unsqueeze(0)  # [B*K*K, W,H]

            x_grid = x_spread.repeat([1, self.width * self.height])
            x_grid = x_grid.reshape([self.num_points, self.width, self.height])
            x_grid = x_grid.unsqueeze(0)  # [B*K*K, W,H]

            y_new = y_center + y_grid
            x_new = x_center + x_grid

            y_new = y_new.repeat(self.num_batch, 1, 1, 1).to(device)
            x_new = x_new.repeat(self.num_batch, 1, 1, 1).to(device)

            y_offset_new = y_offset.detach().clone()

            if if_offset:
                y_offset = y_offset.permute(1, 0, 2, 3)
                y_offset_new = y_offset_new.permute(1, 0, 2, 3)
                center = int(self.num_points // 2)

                # The center position remains unchanged and the rest of the positions begin to swing
                # This part is quite simple. The main idea is that "offset is an iterative process"
                y_offset_new[center] = 0
                for index in range(1, center):
                    y_offset_new[center + index] = (y_offset_new[center + index - 1] + y_offset[center + index])
                    y_offset_new[center - index] = (y_offset_new[center - index + 1] + y_offset[center - index])
                y_offset_new = y_offset_new.permute(1, 0, 2, 3).to(device)
                y_new = y_new.add(y_offset_new.mul(self.extend_scope))

            y_new = y_new.reshape(
                [self.num_batch, self.num_points, 1, self.width, self.height])
            y_new = y_new.permute(0, 3, 1, 4, 2)
            y_new = y_new.reshape([
                self.num_batch, self.num_points * self.width, 1 * self.height
            ])
            x_new = x_new.reshape(
                [self.num_batch, self.num_points, 1, self.width, self.height])
            x_new = x_new.permute(0, 3, 1, 4, 2)
            x_new = x_new.reshape([
                self.num_batch, self.num_points * self.width, 1 * self.height
            ])
            return y_new, x_new

        else:
            """
            Initialize the kernel and flatten the kernel
                y: -num_points//2 ~ num_points//2 (Determined by the kernel size)
                x: only need 0
            """
            y = torch.linspace(
                -int(self.num_points // 2),
                int(self.num_points // 2),
                int(self.num_points),
            )
            x = torch.linspace(0, 0, 1)

            y, x = torch.meshgrid(y, x)
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)

            y_grid = y_spread.repeat([1, self.width * self.height])
            y_grid = y_grid.reshape([self.num_points, self.width, self.height])
            y_grid = y_grid.unsqueeze(0)

            x_grid = x_spread.repeat([1, self.width * self.height])
            x_grid = x_grid.reshape([self.num_points, self.width, self.height])
            x_grid = x_grid.unsqueeze(0)

            y_new = y_center + y_grid
            x_new = x_center + x_grid

            y_new = y_new.repeat(self.num_batch, 1, 1, 1)
            x_new = x_new.repeat(self.num_batch, 1, 1, 1)

            y_new = y_new.to(device)
            x_new = x_new.to(device)
            x_offset_new = x_offset.detach().clone()

            if if_offset:
                x_offset = x_offset.permute(1, 0, 2, 3)
                x_offset_new = x_offset_new.permute(1, 0, 2, 3)
                center = int(self.num_points // 2)
                x_offset_new[center] = 0
                for index in range(1, center):
                    x_offset_new[center + index] = (x_offset_new[center + index - 1] + x_offset[center + index])
                    x_offset_new[center - index] = (x_offset_new[center - index + 1] + x_offset[center - index])
                x_offset_new = x_offset_new.permute(1, 0, 2, 3).to(device)
                x_new = x_new.add(x_offset_new.mul(self.extend_scope))

            y_new = y_new.reshape(
                [self.num_batch, 1, self.num_points, self.width, self.height])
            y_new = y_new.permute(0, 3, 1, 4, 2)
            y_new = y_new.reshape([
                self.num_batch, 1 * self.width, self.num_points * self.height
            ])
            x_new = x_new.reshape(
                [self.num_batch, 1, self.num_points, self.width, self.height])
            x_new = x_new.permute(0, 3, 1, 4, 2)
            x_new = x_new.reshape([
                self.num_batch, 1 * self.width, self.num_points * self.height
            ])
            return y_new, x_new

    """
    input: input feature map [N,C,D,W,H]；coordinate map [N,K*D,K*W,K*H] 
    output: [N,1,K*D,K*W,K*H]  deformed feature map
    """

    def _bilinear_interpolate_3D(self, input_feature, y, x):
        device = input_feature.device
        y = y.reshape([-1]).float()
        x = x.reshape([-1]).float()

        zero = torch.zeros([]).int()
        max_y = self.width - 1
        max_x = self.height - 1

        # find 8 grid locations
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1

        # clip out coordinates exceeding feature map volume
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)
        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)

        input_feature_flat = input_feature.flatten()
        input_feature_flat = input_feature_flat.reshape(
            self.num_batch, self.num_channels, self.width, self.height)
        input_feature_flat = input_feature_flat.permute(0, 2, 3, 1)
        input_feature_flat = input_feature_flat.reshape(-1, self.num_channels)
        dimension = self.height * self.width

        base = torch.arange(self.num_batch) * dimension
        base = base.reshape([-1, 1]).float()

        repeat = torch.ones([self.num_points * self.width * self.height
                             ]).unsqueeze(0)
        repeat = repeat.float()

        base = torch.matmul(base, repeat)
        base = base.reshape([-1])

        base = base.to(device)

        base_y0 = base + y0 * self.height
        base_y1 = base + y1 * self.height

        # top rectangle of the neighbourhood volume
        index_a0 = base_y0 - base + x0
        index_c0 = base_y0 - base + x1

        # bottom rectangle of the neighbourhood volume
        index_a1 = base_y1 - base + x0
        index_c1 = base_y1 - base + x1

        # get 8 grid values
        value_a0 = input_feature_flat[index_a0.type(torch.int64)].to(device)
        value_c0 = input_feature_flat[index_c0.type(torch.int64)].to(device)
        value_a1 = input_feature_flat[index_a1.type(torch.int64)].to(device)
        value_c1 = input_feature_flat[index_c1.type(torch.int64)].to(device)

        # find 8 grid locations
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1

        # clip out coordinates exceeding feature map volume
        y0 = torch.clamp(y0, zero, max_y + 1)
        y1 = torch.clamp(y1, zero, max_y + 1)
        x0 = torch.clamp(x0, zero, max_x + 1)
        x1 = torch.clamp(x1, zero, max_x + 1)

        x0_float = x0.float()
        x1_float = x1.float()
        y0_float = y0.float()
        y1_float = y1.float()

        vol_a0 = ((y1_float - y) * (x1_float - x)).unsqueeze(-1).to(device)
        vol_c0 = ((y1_float - y) * (x - x0_float)).unsqueeze(-1).to(device)
        vol_a1 = ((y - y0_float) * (x1_float - x)).unsqueeze(-1).to(device)
        vol_c1 = ((y - y0_float) * (x - x0_float)).unsqueeze(-1).to(device)

        outputs = (value_a0 * vol_a0 + value_c0 * vol_c0 + value_a1 * vol_a1 +
                   value_c1 * vol_c1)

        if self.morph == 0:
            outputs = outputs.reshape([
                self.num_batch,
                self.num_points * self.width,
                1 * self.height,
                self.num_channels,
            ])
            outputs = outputs.permute(0, 3, 1, 2)
        else:
            outputs = outputs.reshape([
                self.num_batch,
                1 * self.width,
                self.num_points * self.height,
                self.num_channels,
            ])
            outputs = outputs.permute(0, 3, 1, 2)
        return outputs

    def deform_conv(self, input, offset, if_offset):
        y, x = self._coordinate_map_3D(offset, if_offset)
        deformed_feature = self._bilinear_interpolate_3D(input, y, x)
        return deformed_feature

#---------------------------------YOLOv5 专用部分↓---------------------------------
class DSConv_Bottleneck(nn.Module):
    # DSConv bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        self.snc = DSConv(c2, c2, 3, 1, 1, True)

    def forward(self, x):
        return x + self.snc(self.cv2(self.cv1(x))) if self.add else self.snc(self.cv2(self.cv1(x)))


class DSConv_C3(nn.Module):
    # DSConv Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion

        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(DSConv_Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

#---------------------------------YOLOv5 专用部分↑---------------------------------



class SEPC_DSC(nn.Module):
    def __init__(self, in_channels, out_channels, e = 0.5, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(SEPC_DSC, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
        self.out_channels = out_channels
        self.in_channels = in_channels
        c_ = int(out_channels * e)
        self.conv1 = nn.Conv2d(in_channels, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False) #降维 + 残差
        self.conv1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 残差
        self.conv1_3 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 降维

        self.conv2_0 = nn.Conv2d(c_, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
        self.conv2_1 = nn.Conv2d(c_, c_, kernel_size=3, padding=1, stride=1, groups=1, bias=False)
        self.conv2_2 = DSConv(c_, c_, kernel_size=3, extend_scope=1, morph=1, if_offset=True)
        self.conv2_3 = DSConv(c_, c_, kernel_size=5, extend_scope=1, morph=1, if_offset=True)
        self.cat = torch.cat
        self.silu = nn.SiLU()
        self.interpolate = F.interpolate
        self.split = torch.split
        self.bn = nn.BatchNorm2d(128)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def iBN(self, fms, bn):
        sizes = [p.shape[2:] for p in fms]
        n, c = fms[0].shape[0], fms[0].shape[1]
        fm = torch.cat([p.view(n, c, 1, -1) for p in fms], dim=-1)
        fm = self.bn(fm)
        fm = self.split(fm, [s[0] * s[1] for s in sizes], dim=-1)
        return [p.view(n, c, s[0], s[1]) for p, s in zip(fm, sizes)]
    
    def forward(self, x):
        image = [None] * 5
        image1 = [None] * 4
        x = self.conv1(x)
        # print(x.shape)
        image[0] = self.conv2_0(x)
        # print(image[0].shape)
        image[1] = self.conv2_1(x)
        # print(image[1].shape)
        image[2] = self.conv2_2(x)
        # print(image[2].shape)
        image[3] = self.conv2_3(x)
        # print(image[3].shape)
        image[4] = x
        # print(image[4].shape)
        x_out = self.SpatialGate(image[3])
        x_perm1 = image[1].permute(0,2,1,3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        # print(x_out1.shape)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        # print(x_out11.shape)
        x_perm2 = image[2].permute(0,3,2,1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        # print(x_out2.shape)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()
        # print(x_out21.shape)
        image1[0] = x + x_out + x_out21
        # image1[0] = self.conv2_2(x)
        image1[1] = image[0] + x_out11 + x_out21 + x_out
        # image1[1] = self.conv2_2(x)
        image1[2] = x + x_out11 + image[0]
        # image1[2] = self.conv2_2(x)
        image1[3] = x
        # image1[3] = self.conv2_2(x)
        out = self.iBN(image1, nn.BatchNorm2d(self.out_channels))
        out = [self.silu(x.clone()) for x in out]
        # print(out[0].shape)
        # print(out[1].shape)
        out = self.cat([out[0],out[1],out[2],out[3]], 1)
        # print(out.shape)
        out = self.conv1_3(out)
        # print(out.shape)
        out = self.bn1(out)
        # print(out.shape)
        out = self.silu(out)
        # print(out.shape)
        return out
        # x_perm1 = x.permute(0,2,1,3).contiguous()
        # x_out1 = self.ChannelGateH(x_perm1)
        # x_out11 = x_out1.permute(0,2,1,3).contiguous()
        # x_perm2 = x.permute(0,3,2,1).contiguous()
        # x_out2 = self.ChannelGateW(x_perm2)
        # x_out21 = x_out2.permute(0,3,2,1).contiguous()
        # if not self.no_spatial:
        #     x_out = self.SpatialGate(x)
        #     x_out = (1/3)*(x_out + x_out11 + x_out21)
        # else:
        #     x_out = (1/2)*(x_out11 + x_out21)
        # return x_out



class SEPC_DSC_avg(nn.Module):
    def __init__(self, in_channels, out_channels, e = 0.5, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(SEPC_DSC_avg, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
        self.out_channels = out_channels
        self.in_channels = in_channels
        c_ = int(out_channels * e)
        self.conv1 = nn.Conv2d(in_channels, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False) #降维 + 残差
        self.conv1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 残差
        self.conv1_3 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 降维

        self.conv2_0 = nn.Conv2d(c_, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
        self.conv2_1 = nn.Conv2d(c_, c_, kernel_size=3, padding=1, stride=1, groups=1, bias=False)
        self.conv2_2 = DSConv(c_, c_, kernel_size=3, extend_scope=1, morph=1, if_offset=True)
        self.conv2_3 = DSConv(c_, c_, kernel_size=5, extend_scope=1, morph=1, if_offset=True)
        self.cat = torch.cat
        self.silu = nn.SiLU()
        self.interpolate = F.interpolate
        self.split = torch.split
        self.bn = nn.BatchNorm2d(128)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def iBN(self, fms, bn):
        sizes = [p.shape[2:] for p in fms]
        n, c = fms[0].shape[0], fms[0].shape[1]
        fm = torch.cat([p.view(n, c, 1, -1) for p in fms], dim=-1)
        fm = self.bn(fm)
        fm = self.split(fm, [s[0] * s[1] for s in sizes], dim=-1)
        return [p.view(n, c, s[0], s[1]) for p, s in zip(fm, sizes)]
    
    def forward(self, x):
        image = [None] * 5
        image1 = [None] * 4
        x = self.conv1(x)
        # print(x.shape)
        image[0] = self.conv2_0(x)
        # print(image[0].shape)
        image[1] = self.conv2_1(x)
        # print(image[1].shape)
        image[2] = self.conv2_2(x)
        # print(image[2].shape)
        image[3] = self.conv2_3(x)
        # print(image[3].shape)
        image[4] = x
        # print(image[4].shape)
        x_out = self.SpatialGate(image[3])
        x_perm1 = image[1].permute(0,2,1,3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        # print(x_out1.shape)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        # print(x_out11.shape)
        x_perm2 = image[2].permute(0,3,2,1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        # print(x_out2.shape)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()
        # print(x_out21.shape)
        image1[0] = 1/3 * (x + x_out + x_out21)
        # image1[0] = self.conv2_2(x)
        image1[1] = 1/3 * (x_out11 + x_out21 + x_out)
        # image1[1] = self.conv2_2(x)
        image1[2] = 1/3 * (x + x_out11 + x_out)
        # image1[2] = self.conv2_2(x)
        image1[3] = x
        # image1[3] = self.conv2_2(x)
        out = self.iBN(image1, nn.BatchNorm2d(self.out_channels))
        out = [self.silu(x.clone()) for x in out]
        # print(out[0].shape)
        # print(out[1].shape)
        out = self.cat([out[0],out[1],out[2],out[3]], 1)
        # print(out.shape)
        out = self.conv1_3(out)
        # print(out.shape)
        out = self.bn1(out)
        # print(out.shape)
        out = self.silu(out)
        # print(out.shape)
        return out
        # x_perm1 = x.permute(0,2,1,3).contiguous()
        # x_out1 = self.ChannelGateH(x_perm1)
        # x_out11 = x_out1.permute(0,2,1,3).contiguous()
        # x_perm2 = x.permute(0,3,2,1).contiguous()
        # x_out2 = self.ChannelGateW(x_perm2)
        # x_out21 = x_out2.permute(0,3,2,1).contiguous()
        # if not self.no_spatial:
        #     x_out = self.SpatialGate(x)
        #     x_out = (1/3)*(x_out + x_out11 + x_out21)
        # else:
        #     x_out = (1/2)*(x_out11 + x_out21)
        # return x_out





class SEPC_OD(nn.Module):
    def __init__(self, in_channels, out_channels, e = 0.5, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(SEPC_OD, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
        self.out_channels = out_channels
        self.in_channels = in_channels
        c_ = int(out_channels * e)
        self.conv1 = nn.Conv2d(in_channels, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False) #降维 + 残差
        self.conv1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 残差
        self.conv1_3 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=False)  # 降维

        self.conv2_0 = nn.Conv2d(c_, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
        self.conv2_1 = nn.Conv2d(c_, c_, kernel_size=3, padding=1, stride=1, groups=1, bias=False)
        self.conv2_2 = ODConv2d_3rd(c_, c_, kernel_size=3, stride=1, padding=1)
        self.conv2_3 = ODConv2d_3rd(c_, c_, kernel_size=3, stride=1, padding=1)
        self.cat = torch.cat
        self.silu = nn.SiLU()
        self.interpolate = F.interpolate
        self.split = torch.split
        self.bn = nn.BatchNorm2d(128)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def iBN(self, fms, bn):
        sizes = [p.shape[2:] for p in fms]
        n, c = fms[0].shape[0], fms[0].shape[1]
        fm = torch.cat([p.view(n, c, 1, -1) for p in fms], dim=-1)
        fm = self.bn(fm)
        fm = self.split(fm, [s[0] * s[1] for s in sizes], dim=-1)
        return [p.view(n, c, s[0], s[1]) for p, s in zip(fm, sizes)]
    
    def forward(self, x):
        image = [None] * 5
        image1 = [None] * 4
        x = self.conv1(x)
        # print(x.shape)
        image[0] = self.conv2_0(x)
        # print(image[0].shape)
        image[1] = self.conv2_1(x)
        # print(image[1].shape)
        image[2] = self.conv2_2(x)
        # print(image[2].shape)
        image[3] = self.conv2_3(x)
        # print(image[3].shape)
        image[4] = x
        # print(image[4].shape)
        x_out = self.SpatialGate(image[3])
        x_perm1 = image[1].permute(0,2,1,3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        # print(x_out1.shape)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        # print(x_out11.shape)
        x_perm2 = image[2].permute(0,3,2,1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        # print(x_out2.shape)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()
        # print(x_out21.shape)
        image1[0] = x + x_out + x_out21
        # print(image1[0].shape)
        # image1[0] = self.conv2_2(x)
        # print(image1[0].shape)
        image1[1] = image[0] + x_out11 + x_out21 + x_out
        # image1[1] = self.conv2_2(x)
        image1[2] = x + x_out11 + image[0]
        # image1[2] = self.conv2_2(x)
        image1[3] = x
        # image1[3] = self.conv2_2(x)
        out = self.iBN(image1, nn.BatchNorm2d(self.out_channels))
        out = [self.silu(x.clone()) for x in out]
        # print(out[0].shape)
        # print(out[1].shape)
        out = self.cat([out[0],out[1],out[2],out[3]], 1)
        # print(out.shape)
        out = self.conv1_3(out)
        # print(out.shape)
        out = self.bn1(out)
        # print(out.shape)
        out = self.silu(out)
        # print(out.shape)
        return out
        # x_perm1 = x.permute(0,2,1,3).contiguous()
        # x_out1 = self.ChannelGateH(x_perm1)
        # x_out11 = x_out1.permute(0,2,1,3).contiguous()
        # x_perm2 = x.permute(0,3,2,1).contiguous()
        # x_out2 = self.ChannelGateW(x_perm2)
        # x_out21 = x_out2.permute(0,3,2,1).contiguous()
        # if not self.no_spatial:
        #     x_out = self.SpatialGate(x)
        #     x_out = (1/3)*(x_out + x_out11 + x_out21)
        # else:
        #     x_out = (1/2)*(x_out11 + x_out21)
        # return x_out












# import torch
# import torchvision.ops
# from torch import nn
# import math
#
#
# # --------------------------DCNv2 start--------------------------
# class DCNv2(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size=3,
#                  stride=1,
#                  padding=1):
#         super(DCNv2, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride if type(stride) == tuple else (stride, stride)
#         self.padding = padding
#
#         # init weight and bias
#         self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
#         self.bias = nn.Parameter(torch.Tensor(out_channels))
#
#         # offset conv
#         self.conv_offset_mask = nn.Conv2d(in_channels,
#                                           3 * kernel_size * kernel_size,
#                                           kernel_size=kernel_size,
#                                           stride=stride,
#                                           padding=self.padding,
#                                           bias=True)
#
#         # init
#         self.reset_parameters()
#         self._init_weight()
#
#     def reset_parameters(self):
#         n = self.in_channels * (self.kernel_size ** 2)
#         stdv = 1. / math.sqrt(n)
#         self.weight.data.uniform_(-stdv, stdv)
#         self.bias.data.zero_()
#
#     def _init_weight(self):
#         # init offset_mask conv
#         nn.init.constant_(self.conv_offset_mask.weight, 0.)
#         nn.init.constant_(self.conv_offset_mask.bias, 0.)
#
#     def forward(self, x):
#         out = self.conv_offset_mask(x)
#         o1, o2, mask = torch.chunk(out, 3, dim=1)
#         offset = torch.cat((o1, o2), dim=1)
#         mask = torch.sigmoid(mask)
#
#         x = torchvision.ops.deform_conv2d(input=x,
#                                           offset=offset,
#                                           weight=self.weight,
#                                           bias=self.bias,
#                                           padding=self.padding,
#                                           mask=mask,
#                                           stride=self.stride)
#         return x
#
# # ---------------------------DCNv2 end---------------------------
#
#
#
# class SEPC_1(nn.Module): #输入为 20*20*512 小图像深通道应该保持深度
#     def __init__(self, in_channels, out_channels, e = 0.5):
#         super().__init__()
#         self.out_channels = out_channels
#         c_ = int(out_channels * e)
#         self.conv1 = nn.Conv2d(in_channels, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False) #降维 + 残差
#
#
#         self.conv2_0 = nn.Conv2d(c_, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
#         self.conv2_1 = nn.Conv2d(c_, c_, kernel_size=3, padding=1, stride=1, groups=1, bias=False)
#         self.conv2_2 = DCNv2(c_, c_, kernel_size=3, stride=1, padding=1)
#         self.cat = torch.cat
#         self.silu = nn.SiLU()
#         self.interpolate = F.interpolate
#         self.split = torch.split
#         self.bn = nn.BatchNorm2d(128)
#
#     def iBN(self, fms, bn):
#         sizes = [p.shape[2:] for p in fms]
#         n, c = fms[0].shape[0], fms[0].shape[1]
#         fm = torch.cat([p.view(n, c, 1, -1) for p in fms], dim=-1)
#         fm = self.bn(fm)
#         fm = self.split(fm, [s[0] * s[1] for s in sizes], dim=-1)
#         return [p.view(n, c, s[0], s[1]) for p, s in zip(fm, sizes)]
#
#
#     def forward(self, x):
#         image = [None] * 2
#         x = self.conv1(x)
#         # print(x.shape)
#         image[0] = x + self.conv2_2(x)
#         image[1] = self.conv2_1(x) + self.conv2_2(x)
#         out = self.iBN(image, nn.BatchNorm2d(self.out_channels))
#         out = [self.silu(x.clone()) for x in out]
#         out = self.cat([out[0],out[1]], 1)
#         # print(out.shape)
#         return out
#
# class SEPC_2(nn.Module):
#     def __init__(self, in_channels, out_channels, e = 0.5):
#         super().__init__()
#         self.out_channels = out_channels
#         self.in_channels = in_channels
#         c_ = int(out_channels * e)
#         self.conv1 = nn.Conv2d(in_channels, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False) #降维 + 残差
#         self.conv2_0 = nn.Conv2d(c_, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
#         self.conv2_1 = nn.Conv2d(c_, c_, kernel_size=3, padding=1, stride=1, groups=1, bias=False)
#         self.conv2_2 = DCNv2(c_, c_, kernel_size=3, stride=1, padding=1)
#         self.conv2_3 = DCNv2(c_, c_, kernel_size=5, stride=1, padding=2)
#         self.cat = torch.cat
#         self.silu = nn.SiLU()
#         self.interpolate = F.interpolate
#         self.split = torch.split
#         self.bn = nn.BatchNorm2d(128)
#
#     def iBN(self, fms, bn):
#         sizes = [p.shape[2:] for p in fms]
#         n, c = fms[0].shape[0], fms[0].shape[1]
#         fm = torch.cat([p.view(n, c, 1, -1) for p in fms], dim=-1)
#         fm = self.bn(fm)
#         fm = self.split(fm, [s[0] * s[1] for s in sizes], dim=-1)
#         return [p.view(n, c, s[0], s[1]) for p, s in zip(fm, sizes)]
#
#     def forward(self, x):
#         image = [None] * 2
#         x = self.conv1(x)
#         # print(x.shape)
#         image[0] = x + self.conv2_2(x)
#         image[1] = self.conv2_1(x) + self.conv2_2(x) + self.conv2_3(x)
#         out = self.iBN(image, nn.BatchNorm2d(self.out_channels))
#         out = [self.silu(x.clone()) for x in out]
#         # print(out[0].shape)
#         # print(out[1].shape)
#         out = self.cat([out[0],out[1]], 1)
#         # print(out.shape)
#         return out
#
#
#
# class SEPC_12(nn.Module): #输入为 20*20*512 小图像深通道应该保持深度
#     def __init__(self, in_channels, out_channels, e = 0.5):
#         super().__init__()
#         self.out_channels = out_channels
#         c_ = int(out_channels * e)
#         self.conv1 = nn.Conv2d(in_channels, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False) #降维 + 残差
#
#
#         self.conv2_0 = nn.Conv2d(c_, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
#         self.conv2_1 = nn.Conv2d(c_, c_, kernel_size=3, padding=1, stride=1, groups=1, bias=False)
#         self.conv2_2 = DCNv2(c_, c_, kernel_size=3, stride=1, padding=1)
#         self.cat = torch.cat
#         self.silu = nn.SiLU()
#         self.interpolate = F.interpolate
#         self.split = torch.split
#         self.bn = nn.BatchNorm2d(128)
#
#     def iBN(self, fms, bn):
#         sizes = [p.shape[2:] for p in fms]
#         n, c = fms[0].shape[0], fms[0].shape[1]
#         fm = torch.cat([p.view(n, c, 1, -1) for p in fms], dim=-1)
#         fm = self.bn(fm)
#         fm = self.split(fm, [s[0] * s[1] for s in sizes], dim=-1)
#         return [p.view(n, c, s[0], s[1]) for p, s in zip(fm, sizes)]
#
#
#     def forward(self, x):
#         image = [None] * 2
#         x = self.conv1(x)
#         # print(x.shape)
#         image[0] = x + self.conv2_2(x)
#         image[1] = self.conv2_1(x) + self.conv2_2(x) + x
#         out = self.iBN(image, nn.BatchNorm2d(self.out_channels))
#         out = [self.silu(x.clone()) for x in out]
#         out = self.cat([out[0],out[1]], 1)
#         # print(out.shape)
#         return out
#
# class SEPC_22(nn.Module):
#     def __init__(self, in_channels, out_channels, e = 0.5):
#         super().__init__()
#         self.out_channels = out_channels
#         self.in_channels = in_channels
#         c_ = int(out_channels * e)
#         self.conv1 = nn.Conv2d(in_channels, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False) #降维 + 残差
#         self.conv2_0 = nn.Conv2d(c_, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
#         self.conv2_1 = nn.Conv2d(c_, c_, kernel_size=3, padding=1, stride=1, groups=1, bias=False)
#         self.conv2_2 = DCNv2(c_, c_, kernel_size=3, stride=1, padding=1)
#         self.conv2_3 = DCNv2(c_, c_, kernel_size=5, stride=1, padding=2)
#         self.cat = torch.cat
#         self.silu = nn.SiLU()
#         self.interpolate = F.interpolate
#         self.split = torch.split
#         self.bn = nn.BatchNorm2d(128)
#
#     def iBN(self, fms, bn):
#         sizes = [p.shape[2:] for p in fms]
#         n, c = fms[0].shape[0], fms[0].shape[1]
#         fm = torch.cat([p.view(n, c, 1, -1) for p in fms], dim=-1)
#         fm = self.bn(fm)
#         fm = self.split(fm, [s[0] * s[1] for s in sizes], dim=-1)
#         return [p.view(n, c, s[0], s[1]) for p, s in zip(fm, sizes)]
#
#     def forward(self, x):
#         image = [None] * 2
#         x = self.conv1(x)
#         # print(x.shape)
#         image[0] = x + self.conv2_2(x)
#         image[1] = self.conv2_1(x) + self.conv2_2(x) + self.conv2_3(x) + x
#         out = self.iBN(image, nn.BatchNorm2d(self.out_channels))
#         out = [self.silu(x.clone()) for x in out]
#         # print(out[0].shape)
#         # print(out[1].shape)
#         out = self.cat([out[0],out[1]], 1)
#         # print(out.shape)
#         return out
#
#
#
#
# class SEPC_13(nn.Module): #输入为 20*20*512 小图像深通道应该保持深度
#     def __init__(self, in_channels, out_channels, e = 0.5):
#         super().__init__()
#         self.out_channels = out_channels
#         c_ = int(out_channels * e)
#         self.conv1 = nn.Conv2d(in_channels, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False) #降维 + 残差
#         self.conv1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, groups=1, bias=False)  #残差
#         self.conv1_3 = nn.Conv2d(in_channels*2, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
#                                  bias=False)  # 降维
#
#
#         self.conv2_0 = nn.Conv2d(c_, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
#         self.conv2_1 = nn.Conv2d(c_, c_, kernel_size=3, padding=1, stride=1, groups=1, bias=False)
#         self.conv2_2 = DCNv2(c_, c_, kernel_size=3, stride=1, padding=1)
#         self.cat = torch.cat
#         self.silu = nn.SiLU()
#         self.interpolate = F.interpolate
#         self.split = torch.split
#         self.bn = nn.BatchNorm2d(128)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#
#     def iBN(self, fms, bn):
#         sizes = [p.shape[2:] for p in fms]
#         n, c = fms[0].shape[0], fms[0].shape[1]
#         fm = torch.cat([p.view(n, c, 1, -1) for p in fms], dim=-1)
#         fm = self.bn(fm)
#         fm = self.split(fm, [s[0] * s[1] for s in sizes], dim=-1)
#         return [p.view(n, c, s[0], s[1]) for p, s in zip(fm, sizes)]
#
#
#     def forward(self, x):
#         image = [None] * 2
#         y = self.conv1_2(x)
#         # print(y.shape)
#         x = self.conv1(x)
#         # print(x.shape)
#         image[0] = x + self.conv2_2(x)
#         image[1] = self.conv2_1(x) + self.conv2_2(x) + x
#         out = self.iBN(image, nn.BatchNorm2d(self.out_channels))
#         out = [self.silu(x.clone()) for x in out]
#         out = self.cat([out[0],out[1]], 1)
#         # print(out.shape)
#         out = self.cat([out,y],1)
#         # print(out.shape)
#         out = self.conv1_3(out)
#         # print(out.shape)
#         out = self.bn1(out)
#         # print(out.shape)
#         out = self.silu(out)
#         # print(out.shape)
#         return out
#
# class SEPC_23(nn.Module):
#     def __init__(self, in_channels, out_channels, e = 0.5):
#         super().__init__()
#         self.out_channels = out_channels
#         self.in_channels = in_channels
#         c_ = int(out_channels * e)
#         self.conv1 = nn.Conv2d(in_channels, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False) #降维 + 残差
#         self.conv1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
#                                  bias=False)  # 残差
#         self.conv1_3 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
#                                  bias=False)  # 降维
#
#         self.conv2_0 = nn.Conv2d(c_, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
#         self.conv2_1 = nn.Conv2d(c_, c_, kernel_size=3, padding=1, stride=1, groups=1, bias=False)
#         self.conv2_2 = DCNv2(c_, c_, kernel_size=3, stride=1, padding=1)
#         self.conv2_3 = DCNv2(c_, c_, kernel_size=5, stride=1, padding=2)
#         self.cat = torch.cat
#         self.silu = nn.SiLU()
#         self.interpolate = F.interpolate
#         self.split = torch.split
#         self.bn = nn.BatchNorm2d(128)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#
#     def iBN(self, fms, bn):
#         sizes = [p.shape[2:] for p in fms]
#         n, c = fms[0].shape[0], fms[0].shape[1]
#         fm = torch.cat([p.view(n, c, 1, -1) for p in fms], dim=-1)
#         fm = self.bn(fm)
#         fm = self.split(fm, [s[0] * s[1] for s in sizes], dim=-1)
#         return [p.view(n, c, s[0], s[1]) for p, s in zip(fm, sizes)]
#
#     def forward(self, x):
#         image = [None] * 2
#         y = self.conv1_2(x)
#         x = self.conv1(x)
#         # print(x.shape)
#         image[0] = x + self.conv2_2(x)
#         image[1] = self.conv2_1(x) + self.conv2_2(x) + self.conv2_3(x) + x
#         out = self.iBN(image, nn.BatchNorm2d(self.out_channels))
#         out = [self.silu(x.clone()) for x in out]
#         # print(out[0].shape)
#         # print(out[1].shape)
#         out = self.cat([out[0],out[1]], 1)
#         # print(out.shape)
#         out = self.cat([out, y], 1)
#         # print(out.shape)
#         out = self.conv1_3(out)
#         # print(out.shape)
#         out = self.bn1(out)
#         # print(out.shape)
#         out = self.silu(out)
#         # print(out.shape)
#         return out
#
#
#
# class SEPC_14(nn.Module): #输入为 20*20*512 小图像深通道应该保持深度
#     def __init__(self, in_channels, out_channels, e = 0.5):
#         super().__init__()
#         self.out_channels = out_channels
#         c_ = int(out_channels * e)
#         self.conv1 = nn.Conv2d(in_channels, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False) #降维 + 残差
#         self.conv1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, groups=1, bias=False)  #残差
#         self.conv1_3 = nn.Conv2d(in_channels*2, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
#                                  bias=False)  # 降维
#
#
#         self.conv2_0 = nn.Conv2d(c_, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
#         self.conv2_1 = nn.Conv2d(c_, c_, kernel_size=3, padding=1, stride=1, groups=1, bias=False)
#         self.conv2_2 = DCNv2(c_, c_, kernel_size=3, stride=1, padding=1)
#         self.conv2_3 = DCNv2(c_, c_, kernel_size=5, stride=1, padding=2)
#         self.cat = torch.cat
#         self.silu = nn.SiLU()
#         self.interpolate = F.interpolate
#         self.split = torch.split
#         self.bn = nn.BatchNorm2d(128)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#
#     def iBN(self, fms, bn):
#         sizes = [p.shape[2:] for p in fms]
#         n, c = fms[0].shape[0], fms[0].shape[1]
#         fm = torch.cat([p.view(n, c, 1, -1) for p in fms], dim=-1)
#         fm = self.bn(fm)
#         fm = self.split(fm, [s[0] * s[1] for s in sizes], dim=-1)
#         return [p.view(n, c, s[0], s[1]) for p, s in zip(fm, sizes)]
#
#
#     def forward(self, x):
#         image = [None] * 4
#         # print(y.shape)
#         x = self.conv1(x)
#         # print(x.shape)
#         image[0] = x + self.conv2_2(x) + self.conv2_3(x)
#         image[1] = self.conv2_1(x) + self.conv2_2(x) + x + self.conv2_3(x) + self.conv2_0(x)
#         image[2] = self.conv2_0(x) + self.conv2_1(x) + x
#         image[3] = x
#         out = self.iBN(image, nn.BatchNorm2d(self.out_channels))
#         out = [self.silu(x.clone()) for x in out]
#         out = self.cat([out[0],out[1],out[2],out[3]], 1)
#         # print(out.shape)
#         out = self.conv1_3(out)
#         # print(out.shape)
#         out = self.bn1(out)
#         # print(out.shape)
#         out = self.silu(out)
#         # print(out.shape)
#         return out
#
# class SEPC_24(nn.Module):
#     def __init__(self, in_channels, out_channels, e = 0.5):
#         super().__init__()
#         self.out_channels = out_channels
#         self.in_channels = in_channels
#         c_ = int(out_channels * e)
#         self.conv1 = nn.Conv2d(in_channels, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False) #降维 + 残差
#         self.conv1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
#                                  bias=False)  # 残差
#         self.conv1_3 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, padding=0, stride=1, groups=1,
#                                  bias=False)  # 降维
#
#         self.conv2_0 = nn.Conv2d(c_, c_, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
#         self.conv2_1 = nn.Conv2d(c_, c_, kernel_size=3, padding=1, stride=1, groups=1, bias=False)
#         self.conv2_2 = DCNv2(c_, c_, kernel_size=3, stride=1, padding=1)
#         self.conv2_3 = DCNv2(c_, c_, kernel_size=5, stride=1, padding=2)
#         self.cat = torch.cat
#         self.silu = nn.SiLU()
#         self.interpolate = F.interpolate
#         self.split = torch.split
#         self.bn = nn.BatchNorm2d(128)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#
#     def iBN(self, fms, bn):
#         sizes = [p.shape[2:] for p in fms]
#         n, c = fms[0].shape[0], fms[0].shape[1]
#         fm = torch.cat([p.view(n, c, 1, -1) for p in fms], dim=-1)
#         fm = self.bn(fm)
#         fm = self.split(fm, [s[0] * s[1] for s in sizes], dim=-1)
#         return [p.view(n, c, s[0], s[1]) for p, s in zip(fm, sizes)]
#
#     def forward(self, x):
#         image = [None] * 4
#         x = self.conv1(x)
#         # print(x.shape)
#         image[0] = x + self.conv2_2(x) + self.conv2_3(x)
#         image[1] = self.conv2_1(x) + self.conv2_2(x) + self.conv2_3(x) + x + self.conv2_0(x)
#         image[2] = self.conv2_0(x) + self.conv2_1(x) + x
#         image[3] = x
#         out = self.iBN(image, nn.BatchNorm2d(self.out_channels))
#         out = [self.silu(x.clone()) for x in out]
#         # print(out[0].shape)
#         # print(out[1].shape)
#         out = self.cat([out[0],out[1],out[2],out[3]], 1)
#         out = self.conv1_3(out)
#         # print(out.shape)
#         out = self.bn1(out)
#         # print(out.shape)
#         out = self.silu(out)
#         # print(out.shape)
#         return out
    


class BottleneckV10(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))



class Attention(nn.Module):
    def __init__(self, dim, num_heads=8,
                 attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim*2 + self.head_dim, N).split([self.key_dim, self.key_dim, self.head_dim], dim=2)

        attn = (
            (q.transpose(-2, -1) @ k) * self.scale
        )
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x




class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(BottleneckV10(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))





class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )







class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))





class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1, c2, e=0.5):
        """Initializes the PSA module with input/output channels and attention mechanism for feature extraction."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """Executes forward pass in PSA module, applying attention and feed-forward layers to the input tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))




class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))
    


class Silence(nn.Module):
    def __init__(self):
        super(Silence, self).__init__()
    def forward(self, x):    
        return x
    


class RepConvN(nn.Module):
    """RepConv is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process"""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process"""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _avg_to_3x3_tensor(self, avgp):
        channels = self.c1
        groups = self.g
        kernel_size = avgp.kernel_size
        input_dim = channels // groups
        k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
        k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
        return k

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        if hasattr(self, 'conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(in_channels=self.conv1.conv.in_channels,
                              out_channels=self.conv1.conv.out_channels,
                              kernel_size=self.conv1.conv.kernel_size,
                              stride=self.conv1.conv.stride,
                              padding=self.conv1.conv.padding,
                              dilation=self.conv1.conv.dilation,
                              groups=self.conv1.conv.groups,
                              bias=True).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')




class RepNBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConvN(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))




class RepNCSP(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(RepNBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class RepNCSPELAN4(nn.Module):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, c5=1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = c3//2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepNCSP(c3//2, c4, c5), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepNCSP(c4, c4, c5), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3+(2*c4), c2, 1, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))
    


class ELAN1(nn.Module):

    def __init__(self, c1, c2, c3, c4):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = c3//2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3//2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3+(2*c4), c2, 1, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))
    


class AConv(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)
    


class SP(nn.Module):
    def __init__(self, k=3, s=1):
        super(SP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=k // 2)

    def forward(self, x):
        return self.m(x)


class SPPELAN(nn.Module):
    # spp-elan
    def __init__(self, c1, c2, c3):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = SP(5)
        self.cv3 = SP(5)
        self.cv4 = SP(5)
        self.cv5 = Conv(4*c3, c2, 1, 1)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))
    








class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)



class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))





class RepConv(nn.Module):
    # Represented convolution
    # https://arxiv.org/abs/2101.03697

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, deploy=False):
        super(RepConv, self).__init__()

        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2

        assert k == 3
        assert autopad(k, p) == 1

        padding_11 = autopad(k, p) - k // 2

        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)

        else:
            self.rbr_identity = (nn.BatchNorm2d(num_features=c1) if c2 == c1 and s == 1 else None)

            self.rbr_dense = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2),
            )

            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d( c1, c2, 1, s, padding_11, groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2),
            )

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)
    
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return (
            kernel.detach().cpu().numpy(),
            bias.detach().cpu().numpy(),
        )

    def fuse_conv_bn(self, conv, bn):

        std = (bn.running_var + bn.eps).sqrt()
        bias = bn.bias - bn.running_mean * bn.weight / std

        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        weights = conv.weight * t

        bn = nn.Identity()
        conv = nn.Conv2d(in_channels = conv.in_channels,
                              out_channels = conv.out_channels,
                              kernel_size = conv.kernel_size,
                              stride=conv.stride,
                              padding = conv.padding,
                              dilation = conv.dilation,
                              groups = conv.groups,
                              bias = True,
                              padding_mode = conv.padding_mode)

        conv.weight = torch.nn.Parameter(weights)
        conv.bias = torch.nn.Parameter(bias)
        return conv

    def fuse_repvgg_block(self):    
        if self.deploy:
            return
        print(f"RepConv.fuse_repvgg_block")
                
        self.rbr_dense = self.fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])
        
        self.rbr_1x1 = self.fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        rbr_1x1_bias = self.rbr_1x1.bias
        weight_1x1_expanded = torch.nn.functional.pad(self.rbr_1x1.weight, [1, 1, 1, 1])
        
        # Fuse self.rbr_identity
        if (isinstance(self.rbr_identity, nn.BatchNorm2d) or isinstance(self.rbr_identity, nn.modules.batchnorm.SyncBatchNorm)):
            # print(f"fuse: rbr_identity == BatchNorm2d or SyncBatchNorm")
            identity_conv_1x1 = nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=self.groups, 
                    bias=False)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.to(self.rbr_1x1.weight.data.device)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.squeeze().squeeze()
            # print(f" identity_conv_1x1.weight = {identity_conv_1x1.weight.shape}")
            identity_conv_1x1.weight.data.fill_(0.0)
            identity_conv_1x1.weight.data.fill_diagonal_(1.0)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.unsqueeze(2).unsqueeze(3)
            # print(f" identity_conv_1x1.weight = {identity_conv_1x1.weight.shape}")

            identity_conv_1x1 = self.fuse_conv_bn(identity_conv_1x1, self.rbr_identity)
            bias_identity_expanded = identity_conv_1x1.bias
            weight_identity_expanded = torch.nn.functional.pad(identity_conv_1x1.weight, [1, 1, 1, 1])            
        else:
            # print(f"fuse: rbr_identity != BatchNorm2d, rbr_identity = {self.rbr_identity}")
            bias_identity_expanded = torch.nn.Parameter( torch.zeros_like(rbr_1x1_bias) )
            weight_identity_expanded = torch.nn.Parameter( torch.zeros_like(weight_1x1_expanded) )            
        

        #print(f"self.rbr_1x1.weight = {self.rbr_1x1.weight.shape}, ")
        #print(f"weight_1x1_expanded = {weight_1x1_expanded.shape}, ")
        #print(f"self.rbr_dense.weight = {self.rbr_dense.weight.shape}, ")

        self.rbr_dense.weight = torch.nn.Parameter(self.rbr_dense.weight + weight_1x1_expanded + weight_identity_expanded)
        self.rbr_dense.bias = torch.nn.Parameter(self.rbr_dense.bias + rbr_1x1_bias + bias_identity_expanded)
                
        self.rbr_reparam = self.rbr_dense
        self.deploy = True

        if self.rbr_identity is not None:
            del self.rbr_identity
            self.rbr_identity = None

        if self.rbr_1x1 is not None:
            del self.rbr_1x1
            self.rbr_1x1 = None

        if self.rbr_dense is not None:
            del self.rbr_dense
            self.rbr_dense = None









import torch
import torch.nn as nn
 
class SimAM(torch.nn.Module):
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(SimAM, self).__init__()
 
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda
 
    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s
 
    @staticmethod
    def get_module_name():
        return "simam"
 
    def forward(self, x):
 
        b, c, h, w = x.size()
        
        n = w * h - 1
 
        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5
 
        return x * self.activaton(y)
    


class C3_sim(nn.Module):
    # CSP Bottleneck with 3 convolutions
    """在C3TR模块和yolo.py的parse_model模块调用
    CSP Bottleneck with 3 convolutions
    :params c1: 整个BottleneckCSP的输入channel
    :params c2: 整个BottleneckCSP的输出channel
    :params n: 有n个Bottleneck
    :params shortcut: bool Bottleneck中是否有shortcut，默认True
    :params g: Bottleneck中的3x3卷积类型  =1普通卷积  >1深度可分离卷积
    :params e: expansion ratio c2xe=中间其他所有层的卷积核个数/中间所有层的输入输出channel数
    """
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.cv3 = SimAM(c_)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        # 实验性 CrossConv
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv3(self.cv1(x))), self.cv2(x)), 1))
import torch
import torch.nn as nn
import math

from .spr_module import *


__all__ = ['preact_mspanet164', 'preact_mspanet1001']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def convdilated(in_planes, out_planes, kSize=3, stride=1, dilation=1):
    """3x3 convolution with dilation"""
    padding = int((kSize - 1) / 2) * dilation
    return nn.Conv2d(in_planes, out_planes, kernel_size=kSize, stride=stride, padding=padding,
                     dilation=dilation, bias=False)


class MSAModule(nn.Module):
    def __init__(self, inplanes, scale=3, stride=1, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality.
            scale: number of scale.
            stride: conv stride.
            stype: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(MSAModule, self).__init__()

        self.width = inplanes
        self.nums = scale
        self.stride = stride
        assert stype in ['stage', 'normal'], 'One of these is suppported (stage or normal)'
        self.stype = stype

        self.convs = nn.ModuleList([])
        self.bns = nn.ModuleList([])

        for i in range(self.nums):
            if self.stype == 'stage' and self.stride != 1:
                self.convs.append(convdilated(self.width, self.width, stride=stride, dilation=int(i + 1)))
            else:
                self.convs.append(conv3x3(self.width, self.width, stride))

            self.bns.append(nn.BatchNorm2d(self.width))

        self.attention = SPRLayer(self.width)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]

        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i == 0 or (self.stype == 'stage' and self.stride != 1):
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.bns[i](sp)

            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        feats = out
        feats = feats.view(batch_size, self.nums, self.width, feats.shape[2], feats.shape[3])

        sp_inp = torch.split(out, self.width, 1)

        attn_weight = []
        for inp in sp_inp:
            attn_weight.append(self.attention(inp))

        attn_weight = torch.cat(attn_weight, dim=1)
        attn_vectors = attn_weight.view(batch_size, self.nums, self.width, 1, 1)
        attn_vectors = self.softmax(attn_vectors)
        feats_weight = feats * attn_vectors

        for i in range(self.nums):
            x_attn_weight = feats_weight[:, i, :, :, :]
            if i == 0:
                out = x_attn_weight
            else:
                out = torch.cat((out, x_attn_weight), 1)

        return out


class PreMSPABlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=8, scale=3, stype='normal'):
        super(PreMSPABlock, self).__init__()

        width = int(math.floor(planes * (baseWidth / 16.0)))

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(width * scale)
        self.conv2 = MSAModule(width, scale=scale, stride=stride, stype=stype)

        self.bn3 = nn.BatchNorm2d(width * scale)
        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class PreMSPANet(nn.Module):
    def __init__(self, block, layers, num_classes=100, baseWidth=8, scale=3):
        super(PreMSPANet, self).__init__()

        # building first layer
        self.inplanes = 16
        self.baseWidth = baseWidth
        self.scale = scale

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(64 * block.expansion, num_classes)

        # weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            baseWidth=self.baseWidth, scale=self.scale, stype='stage'))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


def preact_mspanet164(**kwargs):

    kwargs['baseWidth'] = 6
    kwargs['scale'] = 3

    model = PreMSPANet(PreMSPABlock, [18, 18, 18], **kwargs)
    return model


def preact_mspanet1001(**kwargs):

    kwargs['baseWidth'] = 8
    kwargs['scale'] = 3

    model = PreMSPANet(PreMSPABlock, [111, 111, 111], **kwargs)
    return model



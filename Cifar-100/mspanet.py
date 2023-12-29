import math
import torch
import torch.nn as nn

from .spr_module import *

__all__ = ['mspanet50', 'mspanet101']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


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


class MSPABlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=30, scale=3, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality.
            planes: output channel dimensionality.
            stride: conv stride.
            downsample: None when stride = 1.
            baseWidth: basic width of conv3x3.
            scale: number of scale.
            stype: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(MSPABlock, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))

        self.conv1 = conv1x1(inplanes, width * scale)
        self.bn1 = nn.BatchNorm2d(width * scale)

        self.conv2 = MSAModule(width, scale=scale, stride=stride, stype=stype)
        self.bn2 = nn.BatchNorm2d(width * scale)

        self.conv3 = conv1x1(width * scale, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class MSPANet(nn.Module):
    def __init__(self, block, layers, num_classes=100, baseWidth=30, scale=3):
        super(MSPANet, self).__init__()

        self.inplanes = 64
        self.baseWidth = baseWidth
        self.scale = scale

        # building first layer
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0],  stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            baseWidth=self.baseWidth, scale=self.scale, stype='stage'))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


def mspanet50(**kwargs):

    model = MSPANet(MSPABlock, [3, 4, 6, 3], baseWidth=30, scale=3, **kwargs)
    return model


def mspanet101(**kwargs):

    model = MSPANet(MSPABlock, [3, 4, 23, 3], baseWidth=30, scale=3, **kwargs)
    return model


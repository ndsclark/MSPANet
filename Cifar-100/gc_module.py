import torch
from torch import nn

__all__ = ['GCModule']


class GCModule(nn.Module):
    def __init__(self, inplanes, ratio=16):
        super(GCModule, self).__init__()

        self.inplanes = inplanes
        self.planes = int(inplanes // ratio)

        self.conv_mask = nn.Conv2d(self.inplanes, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

        self.transform = nn.Sequential(
            nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
            nn.LayerNorm([self.planes, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.planes, self.inplanes, kernel_size=1))

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()

        input_x = x

        input_x = input_x.view(batch, channel, height * width)

        input_x = input_x.unsqueeze(1)

        context_mask = self.conv_mask(x)

        context_mask = context_mask.view(batch, 1, height * width)

        context_mask = self.softmax(context_mask)

        context_mask = context_mask.unsqueeze(-1)

        context = torch.matmul(input_x, context_mask)

        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x):
        context = self.spatial_pool(x)

        attention_term = self.transform(context)
        out = x + attention_term

        return out

import torch
import torch.nn as nn

__all__ = ['DALayer']


class GSA_Module(nn.Module):
    """ Spatial attention module"""
    def __init__(self, in_dim):
        super(GSA_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x

        return out


class GCA_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(GCA_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x

        return out


class DALayer(nn.Module):
    def __init__(self, inp, spatial=True, channel=True):
        """Constructs a DualAttention module.
        Args:
            inp: input channel dimensionality
            spatial: whether to build spatial attention mechanism
            channel: whether to build channel attention mechanism
        """
        super(DALayer, self).__init__()

        if spatial:
            self.s_att = GSA_Module(inp)
        else:
            self.s_att = None

        if channel:
            self.c_att = GCA_Module(inp)
        else:
            self.c_att = None

    def forward(self, x):

        if self.s_att is not None:
            s_out = self.s_att(x)

        if self.c_att is not None:
            c_out = self.c_att(x)

        if (self.s_att is not None) and (self.c_att is not None):
            out = s_out + c_out
        elif self.s_att is not None:
            out = s_out
        elif self.c_att is not None:
            out = c_out
        else:
            assert False, "invalid DALayer"

        return out

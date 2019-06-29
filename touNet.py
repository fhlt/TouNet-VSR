import os
import torch
import torch.nn as nn
from base_networks import *
from rbpn import Net as RBPN

class TouNet(nn.Module):
    def __init__(self, num_channels, base_filter, feat, num_stages, n_resblock, nFrames,
            scale_factor):
        super(TouNet, self).__init__()
        self.forward_rbpn = RBPN(num_channels,base_filter,feat, num_stages=3, n_resblock=5, nFrames=nFrames, scale_factor=scale_factor)
        self.backward_rbpn = RBPN(num_channels,base_filter,feat, num_stages=3, n_resblock=5, nFrames=nFrames, scale_factor=scale_factor)
        # fuse results
        self.output = ConvBlock(num_channels*2, output_size=num_channels,
                kernel_size=3, stride=1, padding=1, bias=True, activation=None, norm=None)

    def forward(self, x, neigbor, flow):
        prediction_forward = self.forward_rbpn(x, neigbor, flow)
        reve_neigbor = neigbor[::-1]
        reve_flow = flow[::-1]
        prediction_backward = self.backward_rbpn(x, reve_neigbor, reve_flow)
        out = torch.cat([prediction_forward,  prediction_backward], 1)
        output = self.output(out)
        return output

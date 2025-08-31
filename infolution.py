'''
Blind spot convolution that weights each photon equally

We termed it "infolution"

Derived from ConvHole used in SUPPORT
Eom, M. et al. Statistically unbiased prediction enables accurate denoising of voltage imaging data. Nature Methods (2023).
https://github.com/NICALab/SUPPORT

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math

from utils import *

class infolution2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(infolution2d, self).__init__()

        # parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # convolution weight

        _w = torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        init.kaiming_uniform_(_w, a=math.sqrt(5))
        self.weight = nn.Parameter(_w, requires_grad=True)
        
        # bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True)
        else:
            self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=False)

    def forward(self, input_list):
        '''
        input_list: a list of frames containing photon arrive times
                    each element is a tensor of size maxPhotonNum * width * height
                    0 if no photon

        in_channels = len(input_list)
        batch, _, width, height = input_list[0].shape
        '''

        lifetime_sum = torch.stack([f.sum(axis=1) for f in input_list], dim=1)
        # intensity = torch.stack([(f>1e-30).float().sum(axis=1) for f in input_list], dim=1)
        intensity = torch.stack([(torch.abs(f)>1e-30).float().sum(axis=1) for f in input_list], dim=1) ## for cos and sin

        o1 = F.conv2d(lifetime_sum, self.weight, stride=self.stride, padding = self.padding, bias = self.bias)
        o2 = F.conv2d(intensity, torch.ones_like(self.weight), stride=self.stride, padding = self.padding)

        o = o1 / (o2 + 1e-8) * self.kernel_size * self.kernel_size # 1e-8 for backward propagation

        o = torch.where(o2 == 0, torch.zeros_like(o), o)
        return o


class infolutionHole2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
    ):
        super(infolutionHole2d, self).__init__()

        # parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # weight
        _w = torch.empty(
            out_channels,
            in_channels,
            self.kernel_size * self.kernel_size - 1,
        )
        init.kaiming_uniform_(_w, a=math.sqrt(5))
        self._oc, self._ic, self._p = _w.size()
        self.weight = nn.Parameter(_w, requires_grad=True)

        # weight: spot
        self._spot = nn.Parameter(torch.zeros([self._oc, self._ic, 1]), requires_grad=False)

        # bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True)
        else:
            self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=False)
    
    def forward(self, input_list):
        '''
        input_list: a list of frames containing photon arrive times
                    each element is a tensor of size maxPhotonNum * width * height
                    0 if no photon

        in_channels = len(input_list)
        batch, _, width, height = input_list[0].shape
        '''

        lifetime = mean_nz(input_list[0], axis=1).unsqueeze(dim=1)
        # intensity = (lifetime>1e-30).float()
        intensity = (torch.abs(lifetime)>1e-30).float() ## for cos and sin

        _kernel = torch.cat(
            [self.weight[:, :, : self._p // 2],
            self._spot,
            self.weight[:, :, self._p // 2 :]],
            dim=2
        )
        _kernel = _kernel.view([self._oc, self._ic, self.kernel_size, self.kernel_size])

        o1 = F.conv2d(lifetime, _kernel, stride=self.stride, padding = self.padding, bias = self.bias)

        _kernel_intensity = torch.ones_like(self.weight)
        _kernel_intensity = torch.cat(
            [_kernel_intensity[:, :, : self._p // 2],
            self._spot,
            _kernel_intensity[:, :, self._p // 2 :]],
            dim=2
        )
        _kernel_intensity = _kernel_intensity.view([self._oc, self._ic, self.kernel_size, self.kernel_size])

        o2 = F.conv2d(intensity, _kernel_intensity, stride=self.stride, padding = self.padding)

        o = o1 / (o2 + 1e-8) * self.kernel_size * self.kernel_size

        o = torch.where(o2 == 0, torch.zeros_like(o), o)
        return o

    
if __name__ == "__main__":
    # test infolution2d
    print("testing infolution2d...")

    testobj = infolution2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False)

    input = torch.ones(1, 2, 7, 7)
    input[0,0,0:3,0:3] = 0
    input[0,1,0:3,0:3] = 0
    input[0,0,3,0] = 0
    input[0,0,3,3] = 0
    input[0,0,5,5] = 0
    input[0,1,5,5] = 0
    input_list = [input]

    testobj.weight = nn.Parameter(0.1111111 * torch.ones_like(testobj.weight), requires_grad=False)

    print('kernel:')
    print(testobj.weight)
    print('input:')
    print(input[0])
    print('output:')
    print(testobj(input_list))

    # test infolutionHole2d
    print("testing infolutionHole2d...")

    testobj = infolutionHole2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False)

    input = torch.ones(1, 2, 7, 7)
    input[0,0,0:3,0:3] = 0
    input[0,1,0:3,0:3] = 0
    input[0,0,3,0] = 0
    input[0,0,3,3] = 0
    input[0,0,5,5] = 0
    input[0,1,5,5] = 0
    input_list = [input]

    testobj.weight = nn.Parameter(0.1111111 * torch.ones_like(testobj.weight), requires_grad=False)

    print('kernel:')
    print(testobj.weight)
    print('input:')
    print(input[0])
    print('output:')
    print(testobj(input_list))
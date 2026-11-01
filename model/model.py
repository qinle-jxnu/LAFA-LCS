import math

import cv2
import numpy as np
import torch
import pywt
import torchvision
import torchvision.ops as ops
from torch.nn.init import xavier_uniform_, constant_
from timm.layers import trunc_normal_
from torch import nn
import torch.nn.functional as F
import pytorch_wavelets as wavelets

from functools import partial
from einops.layers.torch import Rearrange
from torch.nn import init
class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DeformableConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        # init weight and bias
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        # offset conv
        self.conv_offset_mask = nn.Conv2d(in_channels, 3 * kernel_size * kernel_size, kernel_size=kernel_size,
                                          stride=stride, padding=self.padding, bias=True)
        # init
        self.reset_parameters()
        self._init_weight()

    def reset_parameters(self):
        n = self.in_channels * (self.kernel_size**2)
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

        x = torchvision.ops.deform_conv2d(input=x, offset=offset,  weight=self.weight, bias=self.bias,
                                          padding=self.padding, mask=mask, stride=self.stride)
        return x
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b, c, -1).transpose(1, 2)
        mu = x.mean(dim=-1, keepdim=True)
        sigma = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
        x = x.transpose(1, 2).view(b, c, h, w)
        return x
class CA(nn.Module):
    def __init__(self, channel, reduction):
        super(CA, self).__init__()
        self.conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.conv(x)
        return x * y
class TSSD(nn.Module):
    def __init__(self, dim):
        super(TSSD,self).__init__()
        self.down1 = nn.Sequential(
            CA(dim, 8),
            nn.Conv2d(dim, dim, kernel_size=2, stride=2),
            nn.GELU(),
        )
        self.down2 = nn.Sequential(
            CA(dim, 8),
            nn.Conv2d(dim, dim, kernel_size=2, stride=2),
            nn.GELU(),
        )
        self.down3 = nn.Sequential(
            CA(dim, 8),
            nn.Conv2d(dim, dim, kernel_size=2, stride=2),
            nn.GELU(),
        )

        self.mix = nn.Sequential(
            DeformableConv2d(in_channels=dim, out_channels=dim),
            nn.GELU(),
        )

        self.res = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        )
        self.conv_out = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1)

        self.up = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(dim, dim, 3, padding=1),  # 额外的卷积层平滑结果
            LayerNorm(dim),
            nn.GELU()
    )

    def forward(self, x, x_as=None):
        # db
        x_down0 = x
        x_down1 = self.down1(x_down0) + x_as[2] if x_as else self.down1(x_down0)
        x_down2 = self.down2(x_down1) + x_as[1] if x_as else self.down2(x_down1)
        x_down3 = self.down3(x_down2) + x_as[0] if x_as else self.down3(x_down2)
        x_up0 = x_down3
        x_up1 = self.mix(self.up(x_up0) + x_down2)
        x_up2 = self.mix(self.up(x_up1) + x_down1)
        x_up3 = self.mix(self.up(x_up2) + x_down0)

        x_out = x_up3 + self.res(x_up3)

        x = self.conv_out(torch.concat([x, x_out], dim=1))
        return x, [x_up0, x_up1, x_up2]
class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.dim = dim
        self.conv_q = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Conv2d(dim,dim,kernel_size=3, padding=1, groups=dim)
        )
        self.conv_k = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        )
        self.conv_v = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        )
        self.dwt = wavelets.DWTForward(J=1, wave='haar', mode='zero')
        self.conv_qh = nn.Conv2d(3*dim, dim, kernel_size=3,padding=1)
        self.conv_kh = nn.Conv2d(3 * dim, dim, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.norm1 = nn.BatchNorm2d(dim)
        self.softmax = nn.Softmax(dim=1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3,padding=1,groups=dim),
            nn.Conv2d(dim, dim, kernel_size=1)
        )
    def forward(self, x):
        b, c, h, w = x.size()
        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)
        v = v.view(b, c, -1)
        q_coeffs = self.dwt(q)
        k_coeffs = self.dwt(k)
        LL_q, (HL_q, LH_q, HH_q) = q_coeffs[0], (q_coeffs[1][0][:, :, _, :, :] for _ in range(3))
        H_q = self.conv_qh(torch.concat((HL_q,LH_q,HH_q), dim=1))
        L_q = LL_q * self.sigmoid(H_q) + LL_q
        L_q = L_q.view(b,c,-1)
        LL_k, (HL_k, LH_k, HH_k) = k_coeffs[0], (k_coeffs[1][0][:, :, _, :, :] for _ in range(3))
        H_k = self.conv_kh(torch.concat((HL_k, LH_k, HH_k), dim=1))
        L_k = LL_k * self.sigmoid(H_k) + LL_k
        L_k = L_k.view(b,c,-1)
        q = torch.nn.functional.normalize(L_q, dim=-1)
        k = torch.nn.functional.normalize(L_k, dim=-1)
        att =torch.matmul(q, k.transpose(1, 2))
        att = self.softmax(att)
        out = torch.matmul(att,v).view(b, c, h, w)
        out = self.dwconv(out)
        return out

def PhiTPhi_fun(x, PhiW):
    temp = F.conv2d(x, PhiW, padding=0, stride=32, bias=None)
    temp = F.conv_transpose2d(temp, PhiW, stride=32)
    return temp

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class IterationModule(nn.Module):
    def __init__(self,dim,block_size,layer):
        super(IterationModule, self).__init__()
        self.dim = dim
        self.block_size = block_size
        self.alpha = nn.Parameter(torch.Tensor([0.5]))
        self.AConv1 =nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=1, padding=0),
            nn.Conv2d(dim,dim,kernel_size=3, padding=1,groups=dim)
        )
        self.conv1 = nn.Conv2d(dim, 1, 1)
        self.cat = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=True)
        self.attn = Attention(dim) if layer%2==0 else None
        self.post = nn.Conv2d(dim,1,3,1,1)
        self.deno = TSSD(dim)

    def forward(self,xk,z,r,x_pre,Phi):
        r = self.alpha*r
        xk = self.AConv1(xk)
        xk = self.cat(torch.cat([xk,z],dim=1))
        m,x_pre = self.deno(xk,x_pre)
        if self.attn is not None:
            m = self.attn(m) + m
        z = m
        x = m
        x = self.conv1(x)
        n = x - self.alpha * PhiTPhi_fun(x, Phi)
        x = r + n
        return x,z,x_pre

class Model(nn.Module):
    def __init__(self,ratio,depth=10,dim=16,block_size=32):
        super(Model,self).__init__()
        self.dim = dim
        self.ratio = ratio
        self.block_size = block_size
        self.depth = depth
        self.measurement = int(1024 * self.ratio)
        self.alpha = nn.Parameter(torch.Tensor([1.0]))
        self.Phi = nn.Parameter(init.xavier_normal_(torch.Tensor(self.measurement, 1024)))
        self.conv_pre = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, 3, stride=1, padding=1, groups=self.dim),
            nn.Conv2d(self.dim, self.dim, 1, stride=1, padding=0),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, 3, stride=2, padding=1, groups=self.dim),
            nn.Conv2d(self.dim, self.dim, 1, stride=1, padding=0),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, 3, stride=2, padding=1, groups=self.dim),
            nn.Conv2d(self.dim, self.dim, 1, stride=1, padding=0),
        )
        self.conv2 = nn.Conv2d(dim, 1, kernel_size=3, padding=1,bias=True)
        self.register_buffer(
            "identity_m",
            torch.eye(self.measurement),
            persistent=False,
        )
        self.iters = nn.ModuleList()
        for _ in range(self.depth):
            self.iters.append(IterationModule(self.dim,block_size,_))
        self.apply(self._init_weights)
        self.DGM_Conv = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=3, padding=1,bias=True),
        )
        self.res1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 16, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv2d(16, 8, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv2d(8, 1, kernel_size=7, padding=3),
            nn.GELU()
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )


    def forward(self,x):
        x = self.res1(x)+x
        Phi = self.Phi.contiguous().view(self.measurement, 1, 32, 32)
        y = F.conv2d(x, Phi, padding=0, stride=32, bias=None)
        x_init = F.conv_transpose2d(y, Phi, stride=self.block_size, padding=0, bias=None)
        x_pre = None
        r = self.alpha * x_init
        xk = x_init
        x = self.DGM_Conv(xk)
        z = x
        x = self.conv2(x)
        n = x - self.alpha * PhiTPhi_fun(x, Phi)
        x = r + n
        for i in range(self.depth):
            x, z,x_pre = self.iters[i](x, z,r,x_pre,Phi)
        x = x + self.res2(x)
        return x
    def orthogonality_loss(self):
        """Penalize deviations from row orthogonality in the sensing matrix."""
        phi = self.Phi.float()
        gram = phi @ phi.transpose(0, 1)
        identity = self.identity_m.to(dtype=gram.dtype)
        return (gram - identity).square().sum() / self.measurement

    @torch.no_grad()
    def orthogonality_error(self):
        """Return the relative Frobenius error of Phi Phi^T from identity."""
        phi = self.Phi.float()
        gram = phi @ phi.transpose(0, 1)
        identity = self.identity_m.to(dtype=gram.dtype)
        error = torch.linalg.matrix_norm(gram - identity, ord="fro")
        reference = torch.linalg.matrix_norm(identity, ord="fro")
        return (error / reference).item()
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(18,1,96,96).to(device)
    model = Model(ratio=0.1, depth=1,dim=16 ,block_size=32).to(device)
    y = model(x)
    print(y.shape)
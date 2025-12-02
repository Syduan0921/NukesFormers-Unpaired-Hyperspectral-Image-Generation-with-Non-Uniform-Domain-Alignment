import math
import warnings

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')

class AWCA(nn.Module):
    def __init__(self, channel, reduction=16):
        super(AWCA, self).__init__()
        self.conv = nn.Conv2d(channel, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.PReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        input_x = x
        input_x = input_x.view(b, c, h*w).unsqueeze(1)

        mask = self.conv(x).view(b, 1, h*w)
        mask = self.softmax(mask).unsqueeze(-1)
        y = torch.matmul(input_x, mask).view(b, c)

        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class NONLocalBlock2D(nn.Module):
    def __init__(self, in_channels, reduction=8, dimension=2, sub_sample=False, bn_layer=False):
        super(NONLocalBlock2D, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = self.in_channels // reduction

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
        if self.inter_channels == 0:
            self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.LayerNorm
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.LayerNorm
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.LayerNorm

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0, bias=False)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0, bias=False),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0, bias=False)
            nn.init.constant_(self.W.weight, 0)
            # nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0, bias=False)
        # self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
        #                    kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        # phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        # f = torch.matmul(theta_x, phi_x)
        f = self.count_cov_second(theta_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def count_cov_second(self, input):
        x = input
        batchSize, dim, M = x.data.shape
        x_mean_band = x.mean(2).view(batchSize, dim, 1).expand(batchSize, dim, M)
        y = (x - x_mean_band).bmm(x.transpose(1, 2)) / M
        return y


class PSNL(nn.Module):
    def __init__(self, channels):
        super(PSNL, self).__init__()
        # nonlocal module
        self.non_local = NONLocalBlock2D(channels)

    def forward(self,x):
        # divide feature map into 4 part
        batch_size, C, H, W = x.shape
        H1 = int(H / 2)
        W1 = int(W / 2)
        nonlocal_feat = torch.zeros_like(x)

        feat_sub_lu = x[:, :, :H1, :W1]
        feat_sub_ld = x[:, :, H1:, :W1]
        feat_sub_ru = x[:, :, :H1, W1:]
        feat_sub_rd = x[:, :, H1:, W1:]

        nonlocal_lu = self.non_local(feat_sub_lu)
        nonlocal_ld = self.non_local(feat_sub_ld)
        nonlocal_ru = self.non_local(feat_sub_ru)
        nonlocal_rd = self.non_local(feat_sub_rd)
        nonlocal_feat[:, :, :H1, :W1] = nonlocal_lu
        nonlocal_feat[:, :, H1:, :W1] = nonlocal_ld
        nonlocal_feat[:, :, :H1, W1:] = nonlocal_ru
        nonlocal_feat[:, :, H1:, W1:] = nonlocal_rd

        return nonlocal_feat


class Conv3x3(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, dilation=1):
        super(Conv3x3, self).__init__()
        reflect_padding = int(dilation * (kernel_size - 1) / 2)
        self.reflection_pad = nn.ReflectionPad2d(reflect_padding)
        self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride, dilation=dilation, bias=False)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class DRAB(nn.Module):
    def __init__(self, in_dim, out_dim, res_dim, k1_size=3, k2_size=1, dilation=1):
        super(DRAB, self).__init__()
        self.conv1 = Conv3x3(in_dim, in_dim, 3, 1)
        self.relu1 = nn.PReLU()
        self.conv2 = Conv3x3(in_dim, in_dim, 3, 1)
        self.relu2 = nn.PReLU()
        # T^{l}_{1}: (conv.)
        self.up_conv = Conv3x3(in_dim, res_dim, kernel_size=k1_size, stride=1, dilation=dilation)
        self.up_relu = nn.PReLU()
        self.se = AWCA(res_dim)
        # T^{l}_{2}: (conv.)
        self.down_conv = Conv3x3(res_dim, out_dim, kernel_size=k2_size, stride=1)
        self.down_relu = nn.PReLU()

    def forward(self, x, res):
        x_r = x
        x = self.relu1(self.conv1(x))
        x = self.conv2(x)
        x += x_r
        x = self.relu2(x)
        # T^{l}_{1}
        x = self.up_conv(x)
        x += res
        x = self.up_relu(x)
        res = x
        x = self.se(x)
        # T^{l}_{2}
        x = self.down_conv(x)
        x += x_r
        x = self.down_relu(x)
        return x, res


class AWAN(nn.Module):
    def __init__(self, inplanes=3, planes=31, channels=32, n_DRBs=4):
        super(AWAN, self).__init__()
        # 2D Nets
        self.input_conv2D = Conv3x3(inplanes, channels, 3, 1)
        self.input_prelu2D = nn.LeakyReLU()
        self.head_conv2D = Conv3x3(channels, channels, 3, 1)

        self.backbone = nn.ModuleList(
            [DRAB(in_dim=channels, out_dim=channels, res_dim=channels, k1_size=5, k2_size=3, dilation=1) for _ in
             range(n_DRBs)])

        self.tail_conv2D = Conv3x3(channels, channels, 3, 1)
        self.output_prelu2D = nn.LeakyReLU()
        self.output_conv2D = Conv3x3(channels, planes, 3, 1)
        self.tail_nonlocal = PSNL(planes)

    def forward(self, x, layers=[], encode_only=False):
        if -1 in layers:
            layers.append(len(self.model))
        out = self.DRN2D(x)
        if len(layers) > 0:
            if encode_only:
                # print('encoder only return features')
                return [out] # return intermediate features alone; stop in the last layers
            else:
                return out, [out]
        else:
            return out

    def DRN2D(self, x):
        out = self.input_prelu2D(self.input_conv2D(x))
        out = self.head_conv2D(out)
        residual = out
        res = out

        for i, block in enumerate(self.backbone):
            out, res = block(out, res)

        out = self.tail_conv2D(out)
        out = torch.add(out, residual)
        out = self.output_conv2D(self.output_prelu2D(out))
        out = self.tail_nonlocal(out)
        return out
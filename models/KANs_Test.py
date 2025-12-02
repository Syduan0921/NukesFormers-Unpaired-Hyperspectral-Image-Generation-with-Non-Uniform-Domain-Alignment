import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from .KAN_libs import *
from .MST_Plus_Plus import MST


def create_gabor_filter(M, nScale, kernel_size, sigma, theta, dim):
    filters = []
    for scale in range(nScale):
        for direction in range(M):
            # Gabor核参数
            freq = 0.1 + 0.6 * scale / (nScale - 1)  # 频率范围[0.1, 0.7]
            freq /= 2.0
            theta_rad = torch.tensor((direction / M) * math.pi)
            x, y = torch.meshgrid(torch.arange(-kernel_size // 2, kernel_size // 2),
                                  torch.arange(-kernel_size // 2, kernel_size // 2))
            x_theta = x * torch.cos(theta_rad) + y * torch.sin(theta_rad)
            y_theta = -x * torch.sin(theta_rad) + y * torch.cos(theta_rad)
            kernel = torch.exp(-(x_theta**2 + (y_theta / freq)**2) / (2.0 * sigma**2))
            filters.append(kernel)
    filter_gabor = torch.stack(filters)
    filter_gabor = filter_gabor.unsqueeze(1)  # 添加通道维度
    gabor_filter_multi_channel = filter_gabor.repeat(1, dim, 1, 1)  # 复制到31个通道
    return gabor_filter_multi_channel

class GaborLayer(nn.Module):
    def __init__(self, M=2, nScale=4, kernel_size=7, sigma=1.0, theta=0.0):
        super(GaborLayer, self).__init__()

        # Initialize the frequencies as learnable parameters
        init_freqs = torch.tensor([0.1 + 0.6 * scale / (nScale - 1) for scale in range(nScale)]) / 2.0
        self.freqs = nn.Parameter(init_freqs)

        self.M = M
        self.nScale = nScale
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.theta = theta

    def forward(self, x):
        filters = []
        for scale in range(self.nScale):
            for direction in range(self.M):
                freq = self.freqs[scale]
                theta_rad = torch.tensor((direction / self.M) * math.pi)
                x_mesh, y_mesh = torch.meshgrid(torch.arange(-self.kernel_size // 2, self.kernel_size // 2),
                                                torch.arange(-self.kernel_size // 2, self.kernel_size // 2))
                x_theta = x_mesh * torch.cos(theta_rad) + y_mesh * torch.sin(theta_rad)
                y_theta = -x_mesh * torch.sin(theta_rad) + y_mesh * torch.cos(theta_rad)
                kernel = torch.exp(-(x_theta ** 2 + (y_theta / freq) ** 2) / (2.0 * self.sigma ** 2))
                filters.append(kernel)

        filter_stack = torch.stack(filters).unsqueeze(1)
        return filter_stack


class OptimizedGaborConv(nn.Module):
    def __init__(self, in_channels, M=1, nScale=4, kernel_size=7, sigma=1.0, theta=0.0):
        super(OptimizedGaborConv, self).__init__()

        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels * M * nScale, kernel_size, groups=in_channels, padding=kernel_size//2)
        gabor_layer = GaborLayer(M, nScale, kernel_size, sigma, theta)
        weight = gabor_layer.forward(None)  # Call forward of GaborLayer to get the tensor

        # Repeat weights across all channels
        self.depthwise_conv.weight.data = weight.repeat(in_channels, 1, 1, 1)

        # Pointwise convolution to average across channels
        self.pointwise_conv = nn.Conv2d(in_channels * M * nScale, in_channels, 1)
        self.GELU = GELU()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.GELU(x)
        x = self.pointwise_conv(x)
        return x

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


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

def conv(in_channels, out_channels, kernel_size, bias=False, padding = 1, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)


def shift_back(inputs,step=2):          # input [bs,28,256,310]  output [bs, 28, 256, 256]
    [bs, nC, row, col] = inputs.shape
    down_sample = 256//row
    step = float(step)/float(down_sample*down_sample)
    out_col = row
    for i in range(nC):
        inputs[:,i,:,:out_col] = \
            inputs[:,i,:,int(step*i):int(step*i)+out_col]
    return inputs[:, :, :, :out_col]

class New_msa(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )

        #self.ratio = nn.Parameter(torch.ones(1))
        self.gaborT = OptimizedGaborConv(in_channels=dim)
        self.dim = dim

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b,h*w,c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp))
        v = v
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.gaborT(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out

class MS_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b,h*w,c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp))
        v = v
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b,h,w,c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out

class NewFeedForward(nn.Module):
    def __init__(self, dim, patch, mult=4):
        super().__init__()
        self.net_c = nn.Sequential(
            KAN([dim, 2, dim]),
            #KANLinear(in_features=dim*1, out_features=dim),
        )
        self.net_p = nn.Conv2d(dim, dim * 2, 3, 1, 1, bias=False, groups=dim)
        self.GELU = GELU()
        self.ReLU = nn.Sigmoid()
        self.ratio = nn.Parameter(torch.ones(1))

    def forward(self, x_in):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.permute(0, 3, 1, 2)
        out_c = self.net_c(x_in.reshape(b*h*w,c))
        out_c = rearrange(out_c, '(b h w) c -> b c h w', b=b, h=h, w=w, c=c).contiguous()
        x_p = self.net_p(x)
        x_p1, x_p2 = torch.chunk(x_p, chunks=2, dim=1)
        out_p = self.ReLU(x_p1) * x_p2
        out = out_c + out_p * self.ratio
        return out.permute(0, 2, 3, 1)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)

class MSAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            num_blocks,
            patch,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                New_msa(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, NewFeedForward(dim=dim, patch=patch))
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out

class KANMST(nn.Module):
    def __init__(self, in_dim=31, out_dim=31, dim=31, stage=2, num_blocks=[2,4,4], patch_size=64):
        super(KANMST, self).__init__()
        self.dim = dim
        self.stage = stage
        self.patch_size = patch_size

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                MSAB(
                    dim=dim_stage, num_blocks=num_blocks[i], dim_head=dim, heads=dim_stage // dim, patch=self.patch_size),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
            ]))
            dim_stage *= 2
            self.patch_size //= 2

        # Bottleneck
        self.bottleneck = MSAB(
            dim=dim_stage, dim_head=dim, heads=dim_stage // dim, num_blocks=num_blocks[-1], patch=self.patch_size)

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.patch_size *= 2
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                MSAB(
                    dim=dim_stage // 2, num_blocks=num_blocks[stage - 1 - i], dim_head=dim,
                    heads=(dim_stage // 2) // dim, patch=self.patch_size),
            ]))
            dim_stage //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """

        # Embedding
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        for (MSAB, FeaDownSample) in self.encoder_layers:
            fea = MSAB(fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        # Bottleneck
        fea = self.bottleneck(fea)

        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.stage-1-i]], dim=1))
            fea = LeWinBlcok(fea)

        # Mapping
        out = self.mapping(fea) + x

        return out

class KANs(nn.Module):
    def __init__(self, in_channels=3, out_channels=31, n_feat=31, stage=3):
        super(KANs, self).__init__()
        self.stage = stage
        self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=3, padding=(3 - 1) // 2,bias=False)
        #self.conv_in = FastKANConvLayer(in_channels, n_feat, kernel_size=3, padding=(3 - 1) // 2,
                                        #bias=False, stride=1, kan_type='BSpline', use_base_update=False)
        #modules_body = [MST(dim=31, stage=2, num_blocks=[4,7,5]) for _ in range(stage)]
        self.body = nn.ModuleList(
            [KANMST(in_dim=n_feat, out_dim=n_feat, dim=n_feat, stage=2, num_blocks=[2, 4, 4]) for _ in
             range(stage)])
        #self.body = nn.Sequential(*modules_body)
        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=(3 - 1) // 2,bias=False)
        #self.conv_out = FastKANConvLayer(n_feat, out_channels, kernel_size=3, padding=(3 - 1) // 2,
                                         #bias=False, stride=1, kan_type='BSpline', use_base_update=False)

    def forward(self, x, layers=[], encode_only=False):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        b, c, h_inp, w_inp = x.shape
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        #x = self.conv_in(x)
        h = x
        if -1 in layers:
            layers.append(len(self.model))
        if len(layers) > 0:
            feats = []
            for i, model in enumerate(self.body):
                h = model(h)
                feats.append(h)
            h = self.conv_out(h)
            h += x
            if encode_only:
                return feats  # return intermediate features alone; stop in the last layers
            else:
                return h[:, :, :h_inp, :w_inp], feats
        else:
            for i, model in enumerate(self.body):
                h = model(h)
            h = self.conv_out(h)
            h += x
            return h[:, :, :h_inp, :w_inp]
















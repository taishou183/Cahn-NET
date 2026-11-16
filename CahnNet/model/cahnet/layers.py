import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward
from einops import rearrange


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x


class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        yL, yH = self.wt(x)
        yH_reshaped = yH[0].reshape(yH[0].shape[0], -1, yH[0].shape[3], yH[0].shape[4])
        x = torch.cat([yL, yH_reshaped], dim=1)
        x = self.conv_bn_relu(x)
        return x


## UpSampling
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class EUCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(EUCB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)  # 逐点卷积
        )

    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels // 2)
        x = self.pwc(x)
        return x


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        # self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight


class WithBias_DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_DyT(dim)
        else:
            self.body = WithBias_DyT(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class CahnAG(nn.Module):

    def __init__(self, dim=3):
        super(CahnAG, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type='BiasFree')
        self.norm2 = LayerNorm(dim, LayerNorm_type='BiasFree')

        self.Hc1_function = Hc1_function(dim)
        self.He1_function = He1_function(dim)

        self.conv_x = nn.Conv2d(dim, dim, kernel_size=(3, 1), padding=(1, 0), bias=True, groups=dim)
        self.conv_y = nn.Conv2d(dim, dim, kernel_size=(1, 3), padding=(0, 1), bias=True, groups=dim)

        self.gated_mechanism_x = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.gated_mechanism_y = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.proj = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.lam = nn.Parameter(torch.tensor(0.1))

    def _step(self, phi_now, phi_next):
        phi_next = self.norm1(phi_next)
        phi_now = self.norm2(phi_now)

        Hc1 = self.Hc1_function(phi_now)
        # Hc1 = phi_now
        He1 = self.He1_function(phi_next)
        # He1 = phi_next

        uhalf = self.conv_x(Hc1 + He1)
        gate_x = self.gated_mechanism_x(uhalf)

        vhalf = self.conv_y(Hc1 + He1)
        gate_y = self.gated_mechanism_y(vhalf)

        Fxhalf = phi_next * gate_x
        Fyhalf = phi_next * gate_y

        delta = self.proj(Fxhalf + Fyhalf)

        return delta

    def forward(self, phi_next, phi_now):
        delta1 = self._step(phi_now, phi_next)
        phi_pred = phi_now + delta1

        delta2 = self._step(phi_now, phi_pred)

        phi_next = self.lam * (delta1 + delta2) + phi_now
        return phi_next


class Hc1_function(nn.Module):
    def __init__(self, dim, r=8):
        super().__init__()

        # 5×5 深度可分离：DW 5×5 + PW 1×1
        self.branch5 = nn.Sequential(
            nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.SiLU()
        )

        # 3×3 深度可分离：DW 3×3 + PW 1×1
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.SiLU()
        )

        # 1×1 点卷积
        self.branch1 = nn.Conv2d(dim, dim, 1, bias=False)

        self.fuse = nn.Conv2d(3 * dim, dim, 1)

        # 轻量注意力：通道 + 空间
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, 1),
            nn.SiLU(),
            nn.Conv2d(dim // 8, dim, 1),
            nn.Sigmoid()
        )
        self.spat = nn.Sequential(
            nn.Conv2d(dim, dim, 7, padding=3, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.SiLU()
        )

    def forward(self, phi):
        x = phi

        # 三条分支
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)

        x = self.fuse(torch.cat([b1, b3, b5], dim=1))

        # 注意力
        x = x * self.att(x)  # 通道
        x = x * torch.sigmoid(self.spat(x))  # 空间

        # 残差连接
        return phi * x + phi


class He1_function(nn.Module):
    def __init__(self, dim, r=8):
        super().__init__()

        # 5×5 深度可分离：DW 5×5 + PW 1×1
        self.conv5 = nn.Sequential(
            nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.SiLU()
        )

        # 3×3 深度可分离：DW 3×3 + PW 1×1
        self.conv3 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.SiLU()
        )

        # 1×1 点卷积
        self.conv1 = nn.Conv2d(dim, dim, 1, bias=False)

        # 轻量注意力：通道 + 空间
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, 1),
            nn.SiLU(),
            nn.Conv2d(dim // 8, dim, 1),
            nn.Sigmoid()
        )
        self.spat = nn.Sequential(
            nn.Conv2d(dim, dim, 7, padding=3, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.SiLU()
        )

    def forward(self, phi):
        x = phi

        # 5 → 3 → 1
        x = self.conv5(x) + x  # 残差短接
        x = self.conv3(x) + x
        x = self.conv1(x)

        # 注意力
        x = x * self.att(x)  # 通道
        x = x * torch.sigmoid(self.spat(x))  # 空间

        return phi * x + phi

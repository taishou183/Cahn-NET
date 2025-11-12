import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *
from thop import profile


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)
        self.catten = CahnAG(out_channel)

    def forward(self, x):
        x_1 = self.layers(x)
        res = self.catten(x_1, x)
        return res


class DBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)
        self.catten = CahnAG(out_channel)

    def forward(self, x):
        x_1 = self.layers(x)
        res = self.catten(x_1, x)
        return res


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane // 4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_plane, affine=True)
        )

    def forward(self, x):
        x = self.main(x)
        return x


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel * 2, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))


class CahnNet(nn.Module):
    def __init__(self, num_res=6):
        super(CahnNet, self).__init__()

        base_channel = 32

        self.patch_embed = nn.Conv2d(3, base_channel, 3, padding=1)
        self.out_put = nn.Conv2d(base_channel, 3, 3, padding=1)

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel * 2, num_res),
            EBlock(base_channel * 4, num_res),
        ])

        self.sampling = nn.ModuleList([
            BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel * 2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.reduce = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()
        # left_1
        x_in = self.patch_embed(x)
        left_1_out = self.Encoder[0](x_in)
        # left_2
        z = self.sampling[0](left_1_out)
        z = self.FAM2(z, z2)
        left_2_out = self.Encoder[1](z)
        # left_3
        z = self.sampling[1](left_2_out)
        z = self.FAM1(z, z4)
        left_3_out = self.Encoder[2](z)

        # right_3
        right_3_out = self.Decoder[0](left_3_out)
        z_1 = self.ConvsOut[0](right_3_out)
        outputs.append(z_1 + x_4)
        # right_2
        z = self.sampling[2](right_3_out)
        z = self.reduce[0](torch.cat([z, left_2_out], dim=1))
        right_2_out = self.Decoder[1](z)
        z_2 = self.ConvsOut[1](right_2_out)
        outputs.append(z_2 + x_2)
        # right_1
        z = self.sampling[3](right_2_out)
        z = self.reduce[1](torch.cat([z, left_1_out], dim=1))
        right_1_out = self.Decoder[2](z)

        z_3 = self.out_put(right_1_out)
        outputs.append(z_3 + x)

        return outputs


import time


def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 3, 256, 256).to(device)
    net = CahnNet().to(device)
    net.eval()
    print("params", count_param(net))

    flops, params = profile(net, inputs=(x,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
    out = net(x)
    out = net(x)

    times = []

    for i in range(30):
        if 'out' in locals():
            del out
        with torch.no_grad():
            start_time = time.time()
            out = net(x)
            end_time = time.time()
            elapsed_time = end_time - start_time
            times.append(elapsed_time)
            print(f"Run {i + 1}: Time taken = {elapsed_time:.6f} seconds")
            # 清理缓存
        torch.cuda.empty_cache()

    average_time = sum(times) / len(times)
    print(f"Average time over 30 runs: {average_time:.6f} seconds")

    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)

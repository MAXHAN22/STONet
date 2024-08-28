
import torch
from torch import nn
import torch.nn.functional as F

class SAM0(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SAM0, self).__init__()
        # fusion
        self.origin_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            )

        self.Reverse_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            )

        self.out_conv= nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x_origin = self.origin_conv(x)
        x_reverse = self.origin_conv(x)

        mask = 1 - torch.sigmoid(x_origin)

        x_reverse = x_reverse.mul(mask)

        x_out = self.out_conv(x - x_reverse)

        return x_out



class PyramidPooladd1(nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels):
        super(PyramidPooladd1, self).__init__()
        inter_channels = in_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
            )
        self.conv_branch2 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )


        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
            )

        self.conv_branch3 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
            )


        self.conv_branch4 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
            )

        self.conv_branch5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)  # 自适应的平均池化，目标size分别为1x1,2x2,3x3,6x6
        return avgpool(x)

    def upsample(self, x, size):  # 上采样使用双线性插值
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        # branch 1
        x_branch1 = self.upsample(self.conv1(self.pool(x, 6)), size)

        # branch 2
        x_branch2 = self.conv_branch2(x) + x_branch1
        x_branch2 = self.upsample(self.conv2(self.pool(x_branch2, 3)), size)

        # branch 3
        x_branch3 = self.conv_branch3(x) + x_branch2
        x_branch3 = self.upsample(self.conv3(self.pool(x_branch3, 2)), size)

        # branch 4
        x_branch4 = self.conv_branch4(x) + x_branch3
        x_branch4 = self.upsample(self.conv4(self.pool(x_branch4, 1)), size)
        x_out = self.conv_branch5(x) + x_branch4

        return x_out


class PyramidPooladd0(nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels):
        super(PyramidPooladd0, self).__init__()
        inter_channels = in_channels
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            )
        self.conv_branch2 = nn.Conv2d(in_channels, inter_channels, kernel_size=1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            )

        self.conv_branch3 = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            )


        self.conv_branch4 = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            )

        self.conv_branch5 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)  # 自适应的平均池化，目标size分别为1x1,2x2,3x3,6x6
        return avgpool(x)

    def upsample(self, x, size):  # 上采样使用双线性插值
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        # branch 1
        x_branch1 = self.upsample(self.conv1(self.pool(x, 6)), size)

        # branch 2
        x_branch2 = self.relu(self.conv_branch2(x) + x_branch1)
        x_branch2 = self.upsample(self.conv2(self.pool(x_branch2, 3)), size)

        # branch 3
        x_branch3 = self.relu(self.conv_branch3(x) + x_branch2)
        x_branch3 = self.upsample(self.conv3(self.pool(x_branch3, 2)), size)

        # branch 4
        x_branch4 = self.relu(self.conv_branch4(x) + x_branch3)
        x_branch4 = self.upsample(self.conv4(self.pool(x_branch4, 1)), size)
        x_out = self.relu(self.conv_branch5(x) + x_branch4)

        return x_out




class DecoderBlock0(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock0, self).__init__()
        # B, C, H, W -> B, C/2, H, W
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1),
            nn.BatchNorm2d(in_channels // 2),
            nn.LeakyReLU(inplace=True)
        )

        # B, C/2, H, W -> B, C/2, 2H, 2W
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.LeakyReLU(inplace=True)
        )

        # B, C/2, 2H, 2W -> B, C, 2H, 2W
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels // 2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)

        x = self.deconv2(x)

        x = self.conv3(x)

        return x



class decoder2(nn.Module):

    def __init__(self, inchannels, outchannel):
        super(decoder2, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(inchannels, outchannel, 1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(outchannel, outchannel, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )

        inter_channels = outchannel // 2
        self.local_att = nn.Sequential(
            nn.Conv2d(outchannel*2, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, outchannel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(outchannel),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(outchannel*2, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, outchannel, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        x = self.conv1(x)


        x = self.deconv2(x)

        # xa = x + residual
        # xa = x * residual
        xa = torch.cat([x, residual], dim=1)
        xl = self.local_att(xa)
        xg = self.global_att(xa)

        xlg = xl + xg

        wei = self.sigmoid(xlg)

        xo = x * wei + residual * (1 - wei)
        return xo


if __name__ == '__main__':
    rgb = torch.randn(4, 320, 8, 8)
    model = PyramidPooladd0(320, 320)
    a = model(rgb)
    print(a.shape)


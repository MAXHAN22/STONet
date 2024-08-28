import torch
from torch import nn
import torch.nn.functional as F


class SpatialGroupEnhance(nn.Module):
    def __init__(self, groups=4):
        super(SpatialGroupEnhance, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = nn.Parameter(torch.ones(1, groups, 1, 1))
        self.sig = nn.Sigmoid()



    def forward(self, x):
        b, c, h, w = x.shape
        # 通道分组     b * group, c/ group, h ,w
        x = x.view(b * self.groups, -1, h, w)
        # print(self.avg_pool(x).shape)

        # b * group, c / group, h, w
        xn = x * self.avg_pool(x)
        # print(xn.shape)

        #  b * group, 1, h ,w
        xn = xn.sum(dim=1, keepdim = True)
        # print(xn.shape)

        #  b * group, h*w
        t = xn.view(b * self.groups, -1)
        # print(t.shape)

        t = t - t.mean(dim=1, keepdim = True)
        std = t.std(dim=1, keepdim = True) + 1e-5

        t = t / std
        t = t.view(b, self.groups, h, w)
        t = t * self.weight + self.bias
        t = t.view(b * self.groups, 1, h, w)
        # print(x.shape)
        # print(self.sig(t).shape)

        x = x * self.sig(t)
        x = x.view(b, c, h, w)


        return x


class ChannelGroupEnhance(nn.Module):
    def __init__(self, groups=4):
        super(ChannelGroupEnhance, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, groups *groups, 1 ))
        self.bias = nn.Parameter(torch.ones(1, groups *groups, 1 ))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        # HW分组     b * group2, hW/group2, C
        x = x.view(b * self.groups * self.groups, c, -1).permute(0, 2, 1)
        # print(x.shape)
        #  b * group2, HW/group2, C
        xn = x * self.avg_pool(x)
        # print(xn.shape)

        #  b * group2, 1, C
        xn = xn.sum(dim=1, keepdim=True)
        # print(xn.shape)

        # b * group2,  C
        t = xn.view(b * self.groups * self.groups, -1)
        # print(t.shape)

        t = t - t.mean(dim=1, keepdim=True)
        std = t.std(dim=1, keepdim=True) + 1e-5

        t = t / std
        # print(t.shape)
        t = t.view(b, self.groups * self.groups , c)
        t = t * self.weight + self.bias

        t = t.view(b * self.groups * self.groups, 1, c)

        x = x.permute(0,2,1)
        t = t.permute(0,2,1)
        # HW分组     b * group2, hW/group2, C
        # b * group2,  1, c


        x = x * self.sig(t)

        x = x.view(b, c, h, w)

        return x






if __name__ == '__main__':
    rgb = torch.randn(4, 16, 256,256)
    # d = torch.randn(4, 3, 256, 256)
    # model = nn.AdaptiveAvgPool2d(1)
    model = ChannelGroupEnhance()
    #
    a = model(rgb)
    # print(a.shape)
    # rgb = torch.randn(4, 3, 256, 256)
    # d = torch.randn(4, 3, 1, 1)
    # a = rgb * d
    # print(a.shape)
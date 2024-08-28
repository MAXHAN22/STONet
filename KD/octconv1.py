
from torch.nn import functional as F
import torch.nn as nn
import torch

up_kwargs = {'mode': 'nearest'}

class OctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.75, alpha_out=0.75, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, up_kwargs = up_kwargs):
        super(OctaveConv, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = torch.zeros(out_channels).cuda()

        self.up_kwargs = up_kwargs

        # H,W /2
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2,2), stride=2)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out
        self.bn1 = nn.BatchNorm2d(64)
        #
        self.bn2 = nn.BatchNorm2d(192)

    def forward(self, x):
        # 64 64
        X_h ,X_l = x
        end_h_x = int(self.in_channels * (1 - self.alpha_in))
        end_h_y = int(self.out_channels * (1 - self.alpha_out))

        # X_h =  F.conv2d(x, self.weights[0:end_h_y, :, :,:], self.bias[0:end_h_y], 1,
        #                 self.padding, self.dilation, self.groups)
        #
        # X_l =  F.conv2d(x, self.weights[end_h_y:, :, :,:], self.bias[end_h_y:], 1,
        #                 self.padding, self.dilation, self.groups)
        # X_l =  self.h2g_pool(X_l)


        # print(X_h.shape)
        # print(X_h)
        # ([1, 64, 32, 32])
        # print(X_l.shape)
        # print(X_h)
        # ([1, 192, 16, 16])

        X_h2l = self.h2g_pool(X_h)
        # print(X_h2l.shape)
        # ([1, 64, 16, 16])
        # print(X_h2l)

        X_h2h = F.conv2d(X_h, self.weights[0:end_h_y, 0:end_h_x, :,:], self.bias[0:end_h_y], 1,
                        self.padding, self.dilation, self.groups)
        #
        # print(X_h2h.shape)
        # 从([1, 128, 32, 32])
        # print(X_h2h)  全为0 或者 nan 有问题
        X_l2l = F.conv2d(X_l, self.weights[end_h_y:, end_h_x:, :,:], self.bias[end_h_y:], 1,
                        self.padding, self.dilation, self.groups)

        # print(X_l2l.shape)
        # ([1, 384, 16, 16])
        # print(X_l2l) 全为0 或者 nan 有问题
        X_h2l = F.conv2d(X_h2l, self.weights[end_h_y:, 0: end_h_x, :,:], self.bias[end_h_y:], 1,
                        self.padding, self.dilation, self.groups)

        # print(X_h2l.shape)
        # ([1, 384, 16, 16])
        X_l2h = F.conv2d(X_l, self.weights[0:end_h_y, end_h_x:, :,:], self.bias[0:end_h_y], 1,
                        self.padding, self.dilation, self.groups)

        # print(X_l2h.shape)
        # [1, 128, 16, 16])
        X_l2h = F.upsample(X_l2h, scale_factor=2, **self.up_kwargs)

        # print(X_l2h.shape)
        # ([1, 128, 32, 32])
        X_h = X_h2h + X_l2h
        X_l = X_l2l + X_h2l
        #
        # X_h = self.bn1(X_h)
        # X_l = self.bn2(X_l)

        return X_h, X_l




class OctaveConv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.75, stride=1, padding=1):
        super(OctaveConv2, self).__init__()
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.stride = stride
        self.l2l = torch.nn.Conv2d(int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding)
        self.l2h = torch.nn.Conv2d(int(alpha * in_channels), out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding)
        self.h2l = torch.nn.Conv2d(in_channels - int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding)
        self.h2h = torch.nn.Conv2d(in_channels - int(alpha * in_channels),
                                   out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding)

    def forward(self, x):
        X_h, X_l = x



        X_h2l = self.h2g_pool(X_h)

        X_h2h = self.h2h(X_h)
        X_l2h = self.l2h(X_l)

        X_l2l = self.l2l(X_l)
        X_h2l = self.h2l(X_h2l)

        X_l2h = self.upsample(X_l2h)
        X_h = X_l2h + X_h2h
        X_l = X_h2l + X_l2l

        return X_h, X_l


class OctaveConv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.75, stride=1, padding=1):
        super(OctaveConv2, self).__init__()
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.stride = stride
        self.l2l = torch.nn.Conv2d(int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding)
        self.l2h = torch.nn.Conv2d(int(alpha * in_channels), out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding)
        self.h2l = torch.nn.Conv2d(in_channels - int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding)
        self.h2h = torch.nn.Conv2d(in_channels - int(alpha * in_channels),
                                   out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding)

    def forward(self, x):
        X_h, X_l = x[:,:64,:,:,], x[:,64:,:,:,]
        X_l = self.h2g_pool(X_l)

        X_h2l = self.h2g_pool(X_h)

        X_h2h = self.h2h(X_h)
        X_l2h = self.l2h(X_l)

        X_l2l = self.l2l(X_l)
        X_h2l = self.h2l(X_h2l)

        X_l2h = self.upsample(X_l2h)
        X_h = X_l2h + X_h2h
        X_l = X_h2l + X_l2l

        return X_h, X_l


if __name__ == '__main__':
    # # nn.Conv2d
    # # x = torch.randn(1, 256, 32, 32).cuda()
    import torch_dct as DCT
    # x = torch.randn(1, 64, 32, 32).cuda()
    # y = torch.randn(1, 192, 16, 16).cuda()
    # # print(x)
    # # print(y)
    # x = DCT.dct_2d(x, norm='ortho')
    # y = DCT.dct_2d(y, norm='ortho')
    # # print(x)
    # # print(y)
    # # test Oc conv
    # OCconv = OctaveConv2(kernel_size=(3, 3), in_channels=256, out_channels=256, bias=False, stride=2, alpha_in=0.75,
    #                     alpha_out=0.75).cuda()
    # i = x, y
    # x_out, y_out = OCconv(i)
    # print(x_out)
    # print(y_out)
    # nn.Conv2d
    # high = torch.Tensor(1, 64, 32, 32).cuda()
    # low = torch.Tensor(1, 192, 16, 16).cuda()
    x = torch.randn(1, 256, 32, 32).cuda()
    x = DCT.dct_2d(x, norm='ortho')
    # test Oc conv
    OCconv = OctaveConv2(kernel_size=(3, 3), in_channels=256, out_channels=256,  stride=1, alpha=0.75).cuda()

    x_out, y_out = OCconv(x)
    print(x_out)
    print(y_out)



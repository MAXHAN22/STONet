import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.fft
import utils
import cv2
import torch_dct as DCT
""" 给入：（4，3，256，256）
    得到：
    origin_feat_DCT torch.Size([4, 192, 32, 32])
    high torch.Size([4, 96, 32, 32])
    low torch.Size([4, 96, 32, 32])
    
    """
#
# class rgb_to_ycbcr_jpeg(nn.Module):
#     """ Converts RGB image to YCbCr
#     Input:
#         image(tensor): batch x 3 x height x width
#     Output:
#         result(tensor): batch x height x width x 3
#
#     """
#
#     def __init__(self):
#         super(rgb_to_ycbcr_jpeg, self).__init__()
#         matrix = np.array([
#             [0.299, 0.587, 0.114],
#             [-0.168736, -0.331264, 0.5],
#             [0.5, -0.418688, -0.081312]
#         ], dtype=np.float32).T
#
#         self.shift = nn.Parameter(torch.tensor([0., 128., 128.]))
#         self.matrix = nn.Parameter(torch.from_numpy(matrix))
#
#     def forward(self, image):
#         image = image.permute(0, 2, 3, 1)
#         result = torch.tensordot(image, self.matrix, dims=1) + self.shift
#         result.view(image.shape)
#         return result
#
#
# def norm(x):
#     return (1 - torch.exp(-x)) / (1 + torch.exp(-x))
#
# def Seg():
#
#     dict = {0: 0, 1: 1, 2: 8, 3: 16, 4: 9, 5: 2, 6: 3, 7: 10, 8: 17,
#                  9: 24, 10: 32, 11: 25, 12: 18, 13: 11, 14: 4, 15: 5, 16: 12,
#                  17: 19, 18: 26, 19: 33, 20: 40, 21: 48, 22: 41, 23: 34, 24: 27,
#                  25: 20, 26: 13, 27: 6, 28: 7, 29: 14, 30: 21, 31: 28, 32: 35,
#                  33: 42, 34: 49, 35: 56, 36: 57, 37: 50, 38: 43, 39: 36, 40: 29,
#                  41: 22, 42: 15, 43: 23, 44: 30, 45: 37, 46: 44, 47: 51, 48: 58,
#                  49: 59, 50: 52, 51: 45, 52: 38, 53: 31, 54: 39, 55: 46, 56: 53,
#                  57: 60, 58: 61, 59: 54, 60: 47, 61: 55, 62: 62, 63: 63}
#     a = torch.zeros(1, 64, 1, 1)
#
#     for i in range(0, 32):
#         a[0, dict[i+32], 0, 0] = 1
#
#     return a
#
#
# class channel_shuffle(nn.Module):
#     def __init__(self,groups=4):
#         super(channel_shuffle,self).__init__()
#         self.groups=groups
#     def forward(self,x):
#         batchsize, num_channels, height, width = x.data.size()
#         channels_per_group = num_channels // self.groups
#
#         # reshape
#         x = x.view(batchsize, self.groups,
#                channels_per_group, height, width)
#         x = torch.transpose(x, 1, 2).contiguous()
#
#         # flatten
#         x = x.view(batchsize, -1, height, width)
#         return x
#
#
# class Main_Net0(nn.Module):
#
#     def __init__(self):
#         super(Main_Net0, self).__init__()
#         self.rgb_to_ycbcr_jpeg = rgb_to_ycbcr_jpeg()
#
#         self.seg = Seg()
#         self.shuffle = channel_shuffle()
#
#         self.vector_y = nn.Parameter(torch.FloatTensor(1, 64, 1, 1), requires_grad=True)
#         self.vector_cb = nn.Parameter(torch.FloatTensor(1, 64, 1, 1), requires_grad=True)
#         self.vector_cr = nn.Parameter(torch.FloatTensor(1, 64, 1, 1), requires_grad=True)
#
#
#
#     def forward(self, image):
#
#         ycbcr_image = self.rgb_to_ycbcr_jpeg(image)
#
#         num_batchsize = ycbcr_image.shape[0]
#         size = ycbcr_image.shape[2]
#
#         ycbcr_image = ycbcr_image.reshape(num_batchsize, 3, size // 8, 8, size // 8, 8).permute(0, 2, 4, 1, 3, 5)
#         # torch.Size([4, 32, 32, 3, 8, 8])
#         ycbcr_image = DCT.dct_2d(ycbcr_image, norm='ortho')
#         # torch.Size([4, 32, 32, 3, 8, 8])
#         ycbcr_image = ycbcr_image.reshape(num_batchsize, size // 8, size // 8, -1).permute(0, 3, 1, 2)
#         # torch.Size([4, 192, 32, 32])
#         DCT_x = ycbcr_image
#         self.seg = self.seg.to(DCT_x.device)
#         feat_y = DCT_x[:, 0:64, :, :] * (self.seg + norm(self.vector_y))
#         feat_Cb = DCT_x[:, 64:128, :, :] * (self.seg + norm(self.vector_cb))
#         feat_Cr = DCT_x[:, 128:192, :, :] * (self.seg + norm(self.vector_cr))
#
#         # 将三者按channel concat
#         origin_feat_DCT = torch.cat((torch.cat((feat_y, feat_Cb), 1), feat_Cr), 1)
#         #展开
#         origin_feat_DCT = self.shuffle(origin_feat_DCT)
#
#         high = torch.cat([feat_y[:, 32:, :, :], feat_Cb[:, 32:, :, :], feat_Cr[:, 32:, :, :]], 1)
#         low = torch.cat([feat_y[:, :32, :, :], feat_Cb[:, :32, :, :], feat_Cr[:, :32, :, :]], 1)
#         # 已经分为高、低
#
#
#         return origin_feat_DCT, high, low


#
class Main_Net1(nn.Module):

    def __init__(self):
        super(Main_Net1, self).__init__()


    def forward(self, image):
        # num_batchsize = image.shape[0]
        # size = image.shape[2]
        # #
        # image = image.reshape(num_batchsize, 256, size , 1, size , 1).permute(0, 2, 4, 1, 3, 5)
        # image = DCT.dct_2d(image, norm='ortho')
        # image = image.reshape(num_batchsize, size , size , -1).permute(0, 3, 1, 2)
        #
        image = DCT.dct_2d(image, norm='ortho')
        # torch.Size([4, 64, 64, C, 4, 4])




        return image



class Main_Net2(nn.Module):

    def __init__(self):
        super(Main_Net2, self).__init__()


    def forward(self, image):


        dct_am = torch.fft.fft(torch.fft.fft(image, dim=2), dim=3)
        dct_am = torch.abs(torch.real(dct_am))
        # dct_am = torch.real(dct_am)

        # image = DCT.dct_2d(image, norm='ortho')
        # torch.Size([4, 64, 64, C, 4, 4])




        return dct_am



if __name__ == '__main__':
    rgb = torch.randn(4, 256, 32, 32)
    d = torch.randn(4, 256, 256, 256)
    import torch_dct as DCT

    a = DCT.dct_2d(rgb, norm='ortho')
    print(a.shape)


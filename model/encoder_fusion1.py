import torch
import torch.nn as nn
from toolbox.model1.paper1_12.student7_3.student7_3_3.spatial_group_wise_enhancement import SpatialGroupEnhance, ChannelGroupEnhance


class RFEM(nn.Module):
    def __init__(self):
        super(RFEM, self).__init__()
        self.rgb_channel_attention = ChannelGroupEnhance()
        self.rd_spatial_attention = SpatialGroupEnhance()


    def forward(self, r):

        sa = self.rd_spatial_attention(r)
        ca = self.rd_spatial_attention(r)
        # ca = self.rgb_channel_attention(r)
        return sa + ca


class DFEM(nn.Module):
    def __init__(self):
        super(DFEM, self).__init__()
        self.rgb_channel_attention = ChannelGroupEnhance()
        self.rd_spatial_attention = SpatialGroupEnhance()


    def forward(self, d):

        sa = self.rd_spatial_attention(d)
        ca = self.rd_spatial_attention(d)
        # ca = self.rgb_channel_attention(r)
        return sa + ca


#
# changer channel
class fusionmoduleteacher1low(nn.Module):

    def __init__(self, channel_RGB, channel_Depth, channel_inter, channel_out):
        super(fusionmoduleteacher1low, self).__init__()

        self.conv_rgb = nn.Sequential(nn.Conv2d(channel_RGB, channel_inter, 1), nn.BatchNorm2d(channel_inter),
                                      nn.LeakyReLU(inplace=True))
        self.conv_d = nn.Sequential(nn.Conv2d(channel_Depth, channel_inter, 1), nn.BatchNorm2d(channel_inter),
                                    nn.LeakyReLU(inplace=True))

        self.RFEM = RFEM()
        self.DFEM = DFEM()



        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        self.sig = nn.Sigmoid()

        self.conv_final = nn.Sequential(nn.Conv2d(channel_inter, channel_out, 1),
                                        nn.BatchNorm2d(channel_out),
                                        nn.LeakyReLU(inplace=True)
                                        )

        self.sub_conv1 = nn.Sequential(nn.Conv2d(channel_inter, channel_out, 1),
                                       nn.BatchNorm2d(channel_out), nn.LeakyReLU(inplace=True))
        self.sub_conv2 = nn.Sequential(nn.Conv2d(channel_out, channel_out, kernel_size=3, padding=1),
                                       nn.BatchNorm2d(channel_out), nn.LeakyReLU(inplace=True))

    def forward(self, rgb, depth):



        Fr = self.conv_rgb(rgb)
        Fd = self.conv_d(depth)

        Fr = self.RFEM(Fr)
        Fd = self.DFEM(Fd)

        # sub
        sub_feature = self.sub_conv1(torch.abs(Fr - Fd))

        fusion = torch.mul(Fr, Fd)
        # print(fusion.shape)
        max_out, _ = torch.max(fusion, dim=1, keepdim=True)

        fuse_max = self.conv1(max_out)

        fusion_out = torch.mul(fusion, fuse_max)
        fusion_out = self.sig(fusion_out)


        weight_t = fusion_out
        weight_d = 1 - fusion_out

        out_rgb = torch.mul(Fr, weight_t)
        out_d = torch.mul(Fd, weight_d)

        F_out = out_rgb + out_d

        # 恢复c
        F_out = self.conv_final(F_out)

        # 与sub feature相加

        F_out = self.sub_conv2(F_out + sub_feature)

        return F_out



class fusionmoduleteacher0high(nn.Module):
    """Fuse the features from RGB and Depth streams.
    Args:
        channel_RGB: the input channels of RGB stream
        channel_Depth: the input channels of Depth stream
        channel_inter: the channels after first convolution, makes the channels fed into Feature Fusion are same
        channel_out: the output channels of this module
    """

    def __init__(self, channel_RGB, channel_Depth, channel_inter, channel_out):
        super(fusionmoduleteacher0high, self).__init__()

        # channel Reduce
        # self.conv_rgb = nn.Sequential(nn.Conv2d(channel_RGB, channel_inter, 1), nn.BatchNorm2d(channel_inter), nn.LeakyReLU())
        # self.conv_d = nn.Sequential(nn.Conv2d(channel_Depth, channel_inter, 1), nn.BatchNorm2d(channel_inter), nn.LeakyReLU())

        # bconv是因为让特征更smooth
        self.conv_rgb = nn.Sequential(nn.Conv2d(channel_RGB, channel_inter, 1), nn.BatchNorm2d(channel_inter),
                                      nn.LeakyReLU(inplace=True))
        self.conv_d = nn.Sequential(nn.Conv2d(channel_Depth, channel_inter, 1), nn.BatchNorm2d(channel_inter),
                                    nn.LeakyReLU(inplace=True))

        self.RFEM = RFEM()
        self.DFEM = DFEM()

        self.layer_rgb1 = nn.Sequential(nn.Conv2d(channel_inter, channel_inter, kernel_size=3, stride=1, padding=1),
                                        nn.Sigmoid())
        self.layer_d1 = nn.Sequential(nn.Conv2d(channel_inter, channel_inter, kernel_size=3, stride=1, padding=1),
                                      nn.Sigmoid())

        self.layer_rgb2 = nn.Sequential(nn.Conv2d(channel_inter, channel_inter, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(channel_inter), nn.LeakyReLU(inplace=True))
        self.layer_d2 = nn.Sequential(nn.Conv2d(channel_inter, channel_inter, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(channel_inter), nn.LeakyReLU(inplace=True))

        self.conv_3 = nn.Sequential(nn.Conv2d(channel_inter * 2, channel_inter, 1, 1, 0), nn.BatchNorm2d(channel_inter),
                                    nn.LeakyReLU(inplace=True))
        self.conv_4 = nn.Sequential(nn.Conv2d(channel_inter, 1, 1, 1, 0), nn.Sigmoid())

        self.conv_final = nn.Sequential(nn.Conv2d(channel_inter, channel_out, 1),
                                        nn.BatchNorm2d(channel_out),
                                        nn.LeakyReLU(inplace=True)
                                        )

        self.sub_conv1 = nn.Sequential(nn.Conv2d(channel_inter, channel_out, 1),
                                       nn.BatchNorm2d(channel_out), nn.LeakyReLU(inplace=True))
        self.sub_conv2 = nn.Sequential(nn.Conv2d(channel_out, channel_out, kernel_size=3, padding=1),
                                       nn.BatchNorm2d(channel_out), nn.LeakyReLU(inplace=True))

    def forward(self, rgb, depth):


        # 缩小为1/c

        Fr = self.conv_rgb(rgb)
        Fd = self.conv_d(depth)

        Fr = self.RFEM(Fr)
        Fd = self.DFEM(Fd)

        # sub
        sub_feature = self.sub_conv1(torch.abs(Fr - Fd))

        fusion = torch.cat([Fr, Fd], dim=1)



        fusion = self.conv_3(fusion)
        fusion = self.conv_4(fusion)

        # .unsqueeze(1)将第1维删去压缩纬度

        weight_t = fusion
        weight_d = 1 - fusion


        out_rgb = torch.mul(Fr, weight_t)
        out_d = torch.mul(Fd, weight_d)

        F_out = out_rgb + out_d


        # 恢复c
        F_out = self.conv_final(F_out)

        # 与sub feature相加

        F_out = self.sub_conv2(F_out + sub_feature)

        return F_out
#
# class fusionmoduleteacherlowpure(nn.Module):
#
#     def __init__(self, channel_RGB, channel_Depth, channel_inter, channel_out):
#         super(fusionmoduleteacherlowpure, self).__init__()
#
#         self.conv_rgb = nn.Sequential(nn.Conv2d(channel_RGB, channel_inter, 1), nn.BatchNorm2d(channel_inter),
#                                       nn.LeakyReLU(inplace=True))
#         self.conv_d = nn.Sequential(nn.Conv2d(channel_Depth, channel_inter, 1), nn.BatchNorm2d(channel_inter),
#                                     nn.LeakyReLU(inplace=True))
#
#         self.RFEM = RFEM()
#         self.DFEM = DFEM()
#
#
#
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(1, 1, 3, 1, 1),
#             nn.Sigmoid()
#         )
#         self.sig = nn.Sigmoid()
#
#         self.conv_final = nn.Sequential(nn.Conv2d(channel_inter, channel_out, 1),
#                                         nn.BatchNorm2d(channel_out),
#                                         nn.LeakyReLU(inplace=True)
#                                         )
#
#         self.sub_conv1 = nn.Sequential(nn.Conv2d(channel_inter, channel_out, 1),
#                                        nn.BatchNorm2d(channel_out), nn.LeakyReLU(inplace=True))
#         self.sub_conv2 = nn.Sequential(nn.Conv2d(channel_out, channel_out, kernel_size=3, padding=1),
#                                        nn.BatchNorm2d(channel_out), nn.LeakyReLU(inplace=True))
#
#     def forward(self, rgb, depth):
#         # Fr = self.RFEM(rgb)
#         # Fd = self.DFEM(depth)
#
#
#         Fr = self.conv_rgb(rgb)
#         Fd = self.conv_d(depth)
#
#
#
#         # sub
#         sub_feature = self.sub_conv1(torch.abs(Fr - Fd))
#
#         fusion = torch.mul(Fr, Fd)
#         # print(fusion.shape)
#         max_out, _ = torch.max(fusion, dim=1, keepdim=True)
#
#         fuse_max = self.conv1(max_out)
#
#         fusion_out = torch.mul(fusion, fuse_max)
#         fusion_out = self.sig(fusion_out)
#
#
#         weight_t = fusion_out
#         weight_d = 1 - fusion_out
#
#         out_rgb = torch.mul(Fr, weight_t)
#         out_d = torch.mul(Fd, weight_d)
#
#         F_out = out_rgb + out_d
#
#         # 恢复c
#         F_out = self.conv_final(F_out)
#
#         # 与sub feature相加
#
#         F_out = self.sub_conv2(F_out + sub_feature)
#
#         return F_out
#
#
#
# class fusionmoduleteacherhighpure(nn.Module):
#     """Fuse the features from RGB and Depth streams.
#     Args:
#         channel_RGB: the input channels of RGB stream
#         channel_Depth: the input channels of Depth stream
#         channel_inter: the channels after first convolution, makes the channels fed into Feature Fusion are same
#         channel_out: the output channels of this module
#     """
#
#     def __init__(self, channel_RGB, channel_Depth, channel_inter, channel_out):
#         super(fusionmoduleteacherhighpure, self).__init__()
#
#         # channel Reduce
#         # self.conv_rgb = nn.Sequential(nn.Conv2d(channel_RGB, channel_inter, 1), nn.BatchNorm2d(channel_inter), nn.LeakyReLU())
#         # self.conv_d = nn.Sequential(nn.Conv2d(channel_Depth, channel_inter, 1), nn.BatchNorm2d(channel_inter), nn.LeakyReLU())
#
#         # bconv是因为让特征更smooth
#         self.conv_rgb = nn.Sequential(nn.Conv2d(channel_RGB, channel_inter, 1), nn.BatchNorm2d(channel_inter),
#                                       nn.LeakyReLU(inplace=True))
#         self.conv_d = nn.Sequential(nn.Conv2d(channel_Depth, channel_inter, 1), nn.BatchNorm2d(channel_inter),
#                                     nn.LeakyReLU(inplace=True))
#
#         self.RFEM = RFEM()
#         self.DFEM = DFEM()
#
#         self.layer_rgb1 = nn.Sequential(nn.Conv2d(channel_inter, channel_inter, kernel_size=3, stride=1, padding=1),
#                                         nn.Sigmoid())
#         self.layer_d1 = nn.Sequential(nn.Conv2d(channel_inter, channel_inter, kernel_size=3, stride=1, padding=1),
#                                       nn.Sigmoid())
#
#         self.layer_rgb2 = nn.Sequential(nn.Conv2d(channel_inter, channel_inter, kernel_size=3, stride=1, padding=1),
#                                         nn.BatchNorm2d(channel_inter), nn.LeakyReLU(inplace=True))
#         self.layer_d2 = nn.Sequential(nn.Conv2d(channel_inter, channel_inter, kernel_size=3, stride=1, padding=1),
#                                       nn.BatchNorm2d(channel_inter), nn.LeakyReLU(inplace=True))
#
#         self.conv_3 = nn.Sequential(nn.Conv2d(channel_inter * 2, channel_inter, 1, 1, 0), nn.BatchNorm2d(channel_inter),
#                                     nn.LeakyReLU(inplace=True))
#         self.conv_4 = nn.Sequential(nn.Conv2d(channel_inter, 1, 1, 1, 0), nn.Sigmoid())
#
#         self.conv_final = nn.Sequential(nn.Conv2d(channel_inter, channel_out, 1),
#                                         nn.BatchNorm2d(channel_out),
#                                         nn.LeakyReLU(inplace=True)
#                                         )
#
#         self.sub_conv1 = nn.Sequential(nn.Conv2d(channel_inter, channel_out, 1),
#                                        nn.BatchNorm2d(channel_out), nn.LeakyReLU(inplace=True))
#         self.sub_conv2 = nn.Sequential(nn.Conv2d(channel_out, channel_out, kernel_size=3, padding=1),
#                                        nn.BatchNorm2d(channel_out), nn.LeakyReLU(inplace=True))
#
#     def forward(self, rgb, depth):
#
#         # Fr = self.RFEM(rgb)
#         # Fd = self.DFEM(depth)
#         # 缩小为1/c
#
#         Fr = self.conv_rgb(rgb)
#         Fd = self.conv_d(depth)
#
#
#
#         # sub
#         sub_feature = self.sub_conv1(torch.abs(Fr - Fd))
#
#         fusion = torch.cat([Fr, Fd], dim=1)
#
#
#
#         fusion = self.conv_3(fusion)
#         fusion = self.conv_4(fusion)
#
#         # .unsqueeze(1)将第1维删去压缩纬度
#
#         weight_t = fusion
#         weight_d = 1 - fusion
#
#
#         out_rgb = torch.mul(Fr, weight_t)
#         out_d = torch.mul(Fd, weight_d)
#
#         F_out = out_rgb + out_d
#
#
#         # 恢复c
#         F_out = self.conv_final(F_out)
#
#         # 与sub feature相加
#
#         F_out = self.sub_conv2(F_out + sub_feature)
#
#         return F_out
#
#



if __name__ == '__main__':
    rgb = torch.randn(1, 32, 30, 40)
    t = torch.randn(1, 32, 30, 40)
    model = fusionmodulestudent1low(32,16)


    rgb = model(rgb)

    print(rgb.shape)


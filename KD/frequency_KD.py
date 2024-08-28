import itertools
import numpy as np
import torch
import torch.nn as nn
import utils
import cv2
from toolbox.model1.paper1_12.student7_3.student7_3_3.KD.octconv1 import OctaveConv2
from toolbox.model1.paper1_12.student7_3.student7_3_3.KD.transfor import Main_Net1,Main_Net2
from toolbox.model1.paper1_12.student7_3.student7_3_3.KD.KD_loss import *
from toolbox.model1.paper1_12.student7_3.student7_3_3.KD.KD_loss import KLDLoss, hcl, SP, At_loss


class frequency_transfer(nn.Module):

    def __init__(self):
        super(frequency_transfer, self).__init__()

        self.rgb_fre = Main_Net1()
        self.fre_highandlow = OctaveConv2(kernel_size=(3, 3), in_channels=256, out_channels=256,  stride=1, alpha=0.75)

    def forward(self, image):
        X = self.rgb_fre(image)
        X_h, X_l = self.fre_highandlow(X)

        return X_h, X_l

# torch.Size([4, 24, 64, 64])
# torch.Size([4, 32, 32, 32])
# torch.Size([4, 160, 16, 16])
# torch.Size([4, 320, 8, 8])

# class frequency_kd(nn.Module):
#
#
#     def __init__(self):
#         super(frequency_kd, self).__init__()
#         self.frequency_transfer = frequency_transfer()
#
#
#         self.conv4_S = nn.Conv2d(24, 256, 1, 1, 0)
#         self.up4= nn.AvgPool2d(kernel_size=(2,2), stride=2)
#
#         self.conv3_T = nn.Conv2d(512, 256, 1, 1, 0)
#         self.conv3_S = nn.Conv2d(32, 256, 1, 1, 0)
#
#
#         self.conv2_T = nn.Conv2d(1024, 256, 1, 1, 0)
#         self.conv2_S = nn.Conv2d(160, 256, 1, 1, 0)
#         self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#
#         self.conv1_T = nn.Conv2d(2048, 256, 1, 1, 0)
#         self.conv1_S = nn.Conv2d(320, 256, 1, 1, 0)
#         self.up1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
#
#
#         self.softmax = nn.Softmax()
#         self.linear4 = nn.Linear(256*4, 4)
#         self.linear3 = nn.Linear(256*3, 3)
#         self.linear2 = nn.Linear(256*2, 2)
#         self.gap = nn.AdaptiveAvgPool2d(1)
#
#         self.creterion = nn.MSELoss()
#
#     def forward(self, F4, F3, F2, F1, f4, f3, f2, f1):
#         F4= self.up4(F4)
#         f4 = self.up4(self.conv4_S(f4))
#
#         F3 = self.conv3_T(F3)
#         f3 = self.conv3_S(f3)
#
#         F2 = self.up2(self.conv2_T(F2))
#         f2 = self.up2(self.conv2_S(f2))
#
#         F1 = self.up1(self.conv1_T(F1))
#         f1 = self.up1(self.conv1_S(f1))
#
#
        # c4 = torch.cat((F4, F3, F2, F1), dim=1)
        # w4 = self.softmax(self.linear4(self.gap(c4).squeeze(-1).squeeze(-1))).unsqueeze(-1).unsqueeze(-1)
        # w4_1, w4_2, w4_3, w4_4 = w4[:, :1, :, :], w4[:, 1:2, :, :], w4[:, 2:3, :, :], w4[:, 3:4, :, :]
        # out4 = F4 * w4_1 + F3 * w4_2 + F2 * w4_3 + F1 * w4_4
        #
        #
        # c3 = torch.cat((F3, F2, F1), dim=1)
        # w3 = self.softmax(self.linear3(self.gap(c3).squeeze(-1).squeeze(-1))).unsqueeze(-1).unsqueeze(-1)
        # w3_1, w3_2, w3_3 = w3[:, :1, :, :], w3[:, 1:2, :, :], w3[:, 2:3, :, :]
        # out3 = F3 * w3_1 + F2 * w3_2 + F1 * w3_3
        #
        # c2 = torch.cat((F2, F1), dim=1)
        # w2 = self.softmax(self.linear2(self.gap(c2).squeeze(-1).squeeze(-1))).unsqueeze(-1).unsqueeze(-1)
        # w2_1, w2_2= w2[:, :1, :, :], w2[:, 1:2, :, :]
        # out2 =F2 * w2_1 + F1 * w2_2
#
#
#         F4_h, F4_l = self.frequency_transfer(out4)
#         F3_h, F3_l = self.frequency_transfer(out3)
#         F2_h, F2_l = self.frequency_transfer(out2)
#         F1_h, F1_l = self.frequency_transfer(F1)
#
#         f4_h, f4_l = self.frequency_transfer(f4)
#         f3_h, f3_l = self.frequency_transfer(f3)
#         f2_h, f2_l = self.frequency_transfer(f2)
#         f1_h, f1_l = self.frequency_transfer(f1)
#
#         # loss1 = self.creterion(f1_h, F1_h) + self.creterion(f1_l, F1_l)
#         # loss2 = self.creterion(f2_h, F2_h) + self.creterion(f2_l, F2_l)
#         # loss3 = self.creterion(f3_h, F3_h) + self.creterion(f3_l, F3_l)
#         # loss4 = self.creterion(f4_h, F4_h) + self.creterion(f4_l, F4_l)
#
#         loss1 = self.creterion(f1_h, F1_h)
#         loss2 = self.creterion(f2_h, F2_h)
#         loss3 = self.creterion(f3_h, F3_h)
#         loss4 = self.creterion(f4_h, F4_h)
#
#         loss = (loss1 + loss2 + loss3 + loss4) /4
#
#         return loss


class frequency_kd2(nn.Module):


    def __init__(self):
        super(frequency_kd2, self).__init__()
        self.frequency_transfer = frequency_transfer()

        self.conv4_T = nn.Conv2d(256, 256, 3, 2, 1)
        self.conv4_S = nn.Conv2d(24, 256, 3, 2, 1)


        self.conv3_T = nn.Conv2d(512, 256, 1, 1, 0)
        self.conv3_S = nn.Conv2d(32, 256, 1, 1, 0)


        self.conv2_T = nn.Conv2d(1024, 256, 1, 1, 0)
        self.conv2_S = nn.Conv2d(160, 256, 1, 1, 0)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1_T = nn.Conv2d(2048, 256, 1, 1, 0)
        self.conv1_S = nn.Conv2d(320, 256, 1, 1, 0)
        self.up1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)


        # self.softmax = nn.Softmax()
        # self.linear4 = nn.Linear(256*4, 4)
        # self.linear3 = nn.Linear(256*3, 3)
        # self.linear2 = nn.Linear(256*2, 2)
        # self.gap = nn.AdaptiveAvgPool2d(1)

        # self.creterion = KLDLoss()

    def forward(self, F4, F3, F2, F1, f4, f3, f2, f1):
        F4= self.conv4_T(F4)
        f4 = self.conv4_S(f4)

        F3 = self.conv3_T(F3)
        f3 = self.conv3_S(f3)

        F2 = self.up2(self.conv2_T(F2))
        f2 = self.up2(self.conv2_S(f2))

        F1 = self.up1(self.conv1_T(F1))
        f1 = self.up1(self.conv1_S(f1))



        F4_h, F4_l = self.frequency_transfer(F4)
        F3_h, F3_l = self.frequency_transfer(F3)
        F2_h, F2_l = self.frequency_transfer(F2)
        F1_h, F1_l = self.frequency_transfer(F1)

        f4_h, f4_l = self.frequency_transfer(f4)
        f3_h, f3_l = self.frequency_transfer(f3)
        f2_h, f2_l = self.frequency_transfer(f2)
        f1_h, f1_l = self.frequency_transfer(f1)

        # F4_h = F4_h.mean(1)
        # f4_h = f4_h.mean(1)


        # F4_h = F4_h.reshape(F4_h.shape[0], -1)
        # f4_h = f4_h.reshape(f4_h.shape[0], -1)

        # loss = torch.sqrt(torch.sum(torch.pow(F4_h - f4_h, 2), dim=1))

        loss1 = hcl(f1_h, F1_h) + hcl(f1_l, F1_l)
        loss2 = hcl(f2_h, F2_h) + hcl(f2_l, F2_l)
        loss3 = hcl(f3_h, F3_h) + hcl(f3_l, F3_l)
        loss4 = hcl(f4_h, F4_h) + hcl(f4_l, F4_l)

        # t = 2
        # F4_h = F.softmax(F4_h / t, dim=1)
        # F3_h = F.softmax(F3_h / t, dim=1)
        # F2_h = F.softmax(F2_h / t, dim=1)
        # F1_h = F.softmax(F1_h / t, dim=1)
        #
        # F4_l = F.softmax(F4_l / t, dim=1)
        # F3_l = F.softmax(F3_l / t, dim=1)
        # F2_l = F.softmax(F2_l / t, dim=1)
        # F1_l = F.softmax(F1_l / t, dim=1)
        #
        # f4_h = F.softmax(f4_h / t, dim=1)
        # f3_h = F.softmax(f3_h / t, dim=1)
        # f2_h = F.softmax(f2_h / t, dim=1)
        # f1_h = F.softmax(f1_h / t, dim=1)
        #
        # f4_l = F.softmax(f4_l / t, dim=1)
        # f3_l = F.softmax(f3_l / t, dim=1)
        # f2_l = F.softmax(f2_l / t, dim=1)
        # f1_l = F.softmax(f1_l / t, dim=1)
        # #
        # # # torch.Size([1, 64, 32, 32])
        # # # torch.Size([1, 192, 16, 16])
        # #
        # #
        # loss1 = (self.creterion(f1_h, F1_h, F1_h, 6) + self.creterion(f1_l, F1_l, F1_l, 6))* t * t
        # loss2 = (self.creterion(f2_h, F2_h, F1_h, 6) + self.creterion(f2_l, F2_l, F1_l, 6))* t * t
        # loss3 = (self.creterion(f3_h, F3_h, F1_h, 6) + self.creterion(f3_l, F3_l, F1_l, 6))* t * t
        # loss4 = (self.creterion(f4_h, F4_h, F1_h, 6) + self.creterion(f4_l, F4_l, F1_l, 6))* t * t



        loss = (loss1 + loss2 + loss3 + loss4)/4

        return loss

class frequency_kd3(nn.Module):


    def __init__(self):
        super(frequency_kd3, self).__init__()
        self.frequency_transfer = frequency_transfer()

        self.conv4_T = nn.Conv2d(256, 256, 3, 2, 1)
        self.conv4_S = nn.Conv2d(24, 256, 3, 2, 1)


        self.conv3_T = nn.Conv2d(512, 256, 1, 1, 0)
        self.conv3_S = nn.Conv2d(32, 256, 1, 1, 0)


        self.conv2_T = nn.Conv2d(1024, 256, 1, 1, 0)
        self.conv2_S = nn.Conv2d(160, 256, 1, 1, 0)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1_T = nn.Conv2d(2048, 256, 1, 1, 0)
        self.conv1_S = nn.Conv2d(320, 256, 1, 1, 0)
        self.up1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.l1 = nn.SmoothL1Loss()


    def forward(self, F4, F3, F2, F1, f4, f3, f2, f1):
        F4= self.conv4_T(F4)
        f4 = self.conv4_S(f4)

        F3 = self.conv3_T(F3)
        f3 = self.conv3_S(f3)

        F2 = self.up2(self.conv2_T(F2))
        f2 = self.up2(self.conv2_S(f2))

        F1 = self.up1(self.conv1_T(F1))
        f1 = self.up1(self.conv1_S(f1))



        F4_h, F4_l = self.frequency_transfer(F4)
        F3_h, F3_l = self.frequency_transfer(F3)
        F2_h, F2_l = self.frequency_transfer(F2)
        F1_h, F1_l = self.frequency_transfer(F1)

        f4_h, f4_l = self.frequency_transfer(f4)
        f3_h, f3_l = self.frequency_transfer(f3)
        f2_h, f2_l = self.frequency_transfer(f2)
        f1_h, f1_l = self.frequency_transfer(f1)

        loss1 = self.l1(f1_h, F1_h) + self.l1(f1_l, F1_l)
        loss2 = self.l1(f2_h, F2_h) + self.l1(f2_l, F2_l)
        loss3 = self.l1(f3_h, F3_h) + self.l1(f3_l, F3_l)
        loss4 = self.l1(f4_h, F4_h) + self.l1(f4_l, F4_l)





        loss = (loss1 + loss2 + loss3 + loss4)/8

        return loss


if __name__ == '__main__':
    import torch.nn as nn
    import torch
    t1 = torch.randn(4, 256, 64, 64).cuda()
    t2 = torch.randn(4, 512, 32, 32).cuda()
    t3 = torch.randn(4, 1024, 16, 16).cuda()
    t4 = torch.randn(4, 2048, 8, 8).cuda()

    s1 = torch.randn(4, 24, 64, 64).cuda()
    s2 = torch.randn(4, 32, 32, 32).cuda()
    s3 = torch.randn(4, 160, 16, 16).cuda()
    s4 = torch.randn(4, 320, 8, 8).cuda()

    model = frequency_kd3().cuda()
    loss = model(t1,t2,t3,t4,s1,s2,s3,s4)

    print(loss)
    # # print(d)
    # model = frequency_transfer().cuda()
    # rgb1 , rgb2 = model(rgb)
    # d1, d2 = model(d)
    #
    # bn = nn.BatchNorm2d(64).cuda()
    # rgb1 = bn(rgb1)
    # d1 = bn(d1)
    # criterion = nn.MSELoss()
    # rgb1 = rgb1.mean(1).reshape(rgb1.shape[0], -1)
    # d1 = d1.mean(1).reshape(d1.shape[0], -1)
    #
    #
    # destillation_loss = torch.sqrt(torch.sum(torch.pow(rgb1 - d1, 2), dim=1))
    # # print(rgb1)
    # # destillation_loss = torch.sqrt(torch.sum(torch.pow(rgb1 - d1, 2), dim=1))
    #
    # print( destillation_loss )
    # print(rgb1.shape)
    #
    # #
    # # print(a[0].shape)
    # # print(a[2].shape)
    # # print(a[1].shape)

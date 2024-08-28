import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from toolbox.model1.paper1_12.student7.KD.KD_loss import KLDLoss, hcl, SP, At_loss

class feature_kd_loss(nn.Module):
    def __init__(self):
        super(feature_kd_loss, self).__init__()

        self.conv = nn.Conv2d(16, 64, 1, 1, 0)


    def forward(self, R4, D4, rgb4, d4):


        r4 = self.conv(rgb4)
        d4 = self.conv(d4)

        loss_feature_r = hcl(R4, r4)
        loss_feature_d = hcl(D4, d4)

        loss_feature = (loss_feature_r + loss_feature_d) / 2



        return loss_feature

class feature_kd_loss00(nn.Module):
    def __init__(self):
        super(feature_kd_loss00, self).__init__()

        self.conv = nn.Conv2d(16, 64, 1, 1, 0)
        self.SP = SP()

    def forward(self, R4, D4, rgb4, d4):


        r4 = self.conv(rgb4)
        d4 = self.conv(d4)

        loss_feature_r = self.SP(R4, r4)
        loss_feature_d = self.SP(D4, d4)

        loss_feature = (loss_feature_r + loss_feature_d) / 2



        return loss_feature
#
# torch.Size([4, 24, 64, 64])
# torch.Size([4, 32, 32, 32])
# torch.Size([4, 160, 16, 16])
# torch.Size([4, 320, 8, 8])
# 64, 256, 512, 1024, 2048


if __name__ == '__main__':
    r1 = torch.randn(4, 2048, 8, 8)
    d1 = torch.randn(4, 2048, 8, 8)
    r4 = torch.randn(4, 320, 8, 8)
    d4 = torch.randn(4, 320, 8, 8)
    net = feature_kd_loss()
    out = net(r1, d1, r4, d4)

    # 4, 320, 8, 8]) teacher: torch.Size([4, 2048, 8, 8])
    print(out)

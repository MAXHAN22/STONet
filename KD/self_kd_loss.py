import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from toolbox.model1.paper1_12.student7.KD.KD_loss import KLDLoss

def dice_loss(pred, mask):
    mask = torch.sigmoid(mask)
    pred = torch.sigmoid(pred)
    intersection = (pred * mask).sum(axis=(2, 3))
    unior = (pred + mask).sum(axis=(2, 3))
    dice = (2 * intersection + 1) / (unior + 1)
    dice = torch.mean(1 - dice)
    return dice

def hcl(fstudent, fteacher):
    loss_all = 0.0
    B, C, h, w = fstudent.size()
    loss = F.mse_loss(fstudent, fteacher, reduction='mean')
    cnt = 1.0
    tot = 1.0
    for l in [4,2,1]:
        if l >=h:
            continue
        tmpfs = F.adaptive_avg_pool2d(fstudent, (l,l))
        tmpft = F.adaptive_avg_pool2d(fteacher, (l,l))
        cnt /= 2.0
        loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
        tot += cnt
    loss = loss / tot
    loss_all = loss_all + loss
    return loss_all



#
# torch.Size([4, 24, 64, 64])
# torch.Size([4, 32, 32, 32])
# torch.Size([4, 160, 16, 16])
# torch.Size([4, 320, 8, 8])



class self_kd_loss1(nn.Module):
    def __init__(self):
        super(self_kd_loss1, self).__init__()


    def forward(self, F4, F3, F2, F1):

        # 对于decoder3来说，只能用4来kd3


        loss_self_kd3 = hcl(F4, F3)

        # 对于2来说，可以用4或者3
        loss_self_kd4_2 = hcl(F4, F2)

        # 选择loss大的


        # 对于21来说，可以用4,3,2
        loss_self_kd4_1 = hcl(F4, F1)



        loss = (loss_self_kd3 + loss_self_kd4_2 + loss_self_kd4_1) / 3

        return loss







if __name__ == '__main__':
    r1 = torch.randn(4, 24, 64, 64)
    d1 = torch.randn(4, 24, 64, 64)
    r4 = torch.randn(4, 24, 64, 64)
    d4 = torch.randn(4, 24, 64, 64)
    net = self_kd_loss3()
    out = net(r1, d1, r4, d4)


    print(out)

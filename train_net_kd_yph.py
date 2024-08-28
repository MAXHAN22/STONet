import json
import torch
import os
import argparse
from torch import nn
import torch.nn.functional as F
import numpy as np
from toolbox.datasets.vaihingen import Vaihingen
from toolbox.datasets.potsdam import Potsdam
from torch.utils.data import DataLoader
from torch import optim
from datetime import datetime
from torch.autograd import Variable

from toolbox.datasets.potsdam import Potsdam
from toolbox.datasets.vaihingen import Vaihingen

from toolbox.optim.Ranger import Ranger
from toolbox.loss.loss import MscCrossEntropyLoss, FocalLossbyothers, MscLovaszSoftmaxLoss
from toolbox.model1.paper1_12.student7_3_3.KD.KD_loss import KLDLoss, hcl, SP, At_loss,dice_loss

from toolbox.model1.paper1_12.student7_3_3.teacher import  student1 as teacher
from toolbox.model1.paper1_12.student7_3_3.ablation_fusion.studentwithC import student1 as student
# from toolbox.model1.paper1_12.student7_3_3.KD.self_kd_loss import self_kd_loss0
from toolbox.model1.paper1_12.student7_3_3.KD.frequency_KD import frequency_kd2
from toolbox.model1.paper1_12.student7_3_3.KD.feature_kd_loss import feature_kd_loss
# from toolbox.model1.paper1_12.student7_3_3.KD.transfor import Main_Net
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



DATASET = "Potsdam"
# DATASET = "Vaihingen"
batch_size = 16


parser = argparse.ArgumentParser(description="config")
parser.add_argument(
    "--config",
    nargs="?",
    type=str,
    default="configs/{}.json".format(DATASET),
    help="Configuration file to use",
)
args = parser.parse_args()

with open(args.config, 'r') as fp:
    cfg = json.load(fp)
if DATASET == "Potsdam":
    train_dataloader = DataLoader(Potsdam(cfg, mode='train'), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(Potsdam(cfg, mode='test'), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
elif DATASET == "Vaihingen":
    train_dataloader = DataLoader(Vaihingen(cfg, mode='train'), batch_size=batch_size, shuffle=True, num_workers=4,
                                  pin_memory=True)
    test_dataloader = DataLoader(Vaihingen(cfg, mode='test'), batch_size=batch_size, shuffle=True, num_workers=4,
                                 pin_memory=True)


criterion0 = KLDLoss().cuda()
criterion1 = feature_kd_loss().cuda()
criterion2 = At_loss(2).cuda()
# criterion3 = self_kd_loss3().cuda()
criterion4 = frequency_kd2().cuda()
# criterion3 = hcl().cuda()


criterion_without = MscCrossEntropyLoss().cuda()
# criterion_focal1 = FocalLossbyothers().cuda()
# criterion_Lovasz = MscLovaszSoftmaxLoss().cuda()
criterion_bce = nn.BCELoss().cuda()  # 边界监督


net_s = student().cuda()
net_T = teacher().cuda()
net_T.load_state_dict(torch.load('./weight/paper1_1/t_2/1021tp-100-Potsdam-loss.pth'))

# for p in net_T.parameters():
#     p.stop_gradient = True
# net_T.eval()

optimizer = optim.Adam(net_s.parameters(), lr=1e-4, weight_decay=5e-4)


def accuary(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size

best = [0.0]
size = (56, 56)
numloss = 0
nummae = 0
trainlosslist_nju = []
vallosslist_nju = []
iter_num = len(train_dataloader)
epochs = 120
# schduler_lr = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)  # setting the learning rate desend starage
for epoch in range(epochs):
    if epoch % 20 == 0 and epoch != 0:  # setting the learning rate desend starage
        for group in optimizer.param_groups:
            group['lr'] = 0.1 * group['lr']
    # for group in optimizer.param_groups:
    # 	group['lr'] *= 0.99
    # 	print(group['lr'])
    train_loss = 0
    net = net_s.train()
    prec_time = datetime.now()
    alpha = 0.90
    for i, sample in enumerate(train_dataloader):
        image = Variable(sample['image'].cuda())  # [2, 3, 256, 256]
        ndsm = Variable(sample['dsm'].cuda())  # [2, 1, 256, 256]
        label = Variable(sample['label'].long().cuda())  # [2, 256, 256]
        ndsm = torch.repeat_interleave(ndsm, 3, dim=1)
        outs = net(image, ndsm)

        with torch.no_grad():
            outt = net_T(image, ndsm)

        # out, out2, out3, out4, f1, f2, f3, f4, rgb4, d4

        # loss lable  1.8350498676300049

        loss_label0 = criterion_without(outs[0:4], label)

        # loss decoder   用最后一个增强所有
        teacher_out = outt[0].data.cpu().numpy()
        teacher_out = np.argmax(teacher_out, axis=1)
        teacher_out = torch.from_numpy(teacher_out)
        teacher_out = Variable(teacher_out.long().cuda())

        loss0 = criterion_without(outs[0], teacher_out)



        loss_decoder_1 = criterion0(outs[1], outt[0], label, 4)
        loss_decoder_2 = criterion0(outs[2], outt[0], label, 4)
        loss_decoder_3 = criterion0(outs[3], outt[0], label, 4)

        loss1 = loss_decoder = (loss_decoder_3 + loss_decoder_2 + loss_decoder_1 ) / 3


        loss_deocder2_1 = dice_loss(outs[1], outt[0])
        loss_deocder2_2 = dice_loss(outs[2], outt[0])
        loss_deocder2_3 = dice_loss(outs[3], outt[0])
        loss2 = ( loss_deocder2_1 + loss_deocder2_2 + loss_deocder2_3)/3

        # loss encoder fusion
        loss3 = loss_encoder_fusion_kd = criterion4(outt[4], outt[5], outt[6], outt[7], outs[4], outs[5], outs[6], outs[7])


        # loss encoder feature  student:torch.Size([4, 320, 8, 8]) teacher:torch.Size([4, 2048, 8, 8])

        loss4 = loss_feature = criterion1(outt[16], outt[17], outs[16], outs[17])

        # loss_kd = loss_label0 + loss_encoder_fusion_kd + loss_decoder + loss_feature
        #
        #
        # loss = loss_kd + loss_skd

        loss = loss_label0 + loss3 +loss1 + loss2 + loss4 + loss0



        print('Training: Iteration {:4}'.format(i), 'Loss:', loss.item())
        if (i+1) % 100 == 0:
            print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  loss : %5.4f' % (
                epoch+1, epochs, i+1, iter_num, train_loss / 100))
            train_loss = 0

        optimizer.zero_grad()

        loss.backward()  # backpropagation to get gradient
        # qichuangaaaxuexi
        optimizer.step()  # update the weight

        train_loss = loss.item() + train_loss

    net = net_s.eval()
    eval_loss = 0
    acc = 0
    with torch.no_grad():
        for j, sampleTest in enumerate(test_dataloader):
            imageVal = Variable(sampleTest['image'].float().cuda())
            ndsmVal = Variable(sampleTest['dsm'].float().cuda())
            labelVal = Variable(sampleTest['label'].long().cuda())
            ndsmVal = torch.repeat_interleave(ndsmVal, 3, dim=1)
            # imageVal = F.interpolate(imageVal, (320, 320), mode="bilinear", align_corners=True)
            # ndsmVal = F.interpolate(ndsmVal, (320, 320), mode="bilinear", align_corners=True)
            # labelVal = F.interpolate(labelVal.unsqueeze(1).float(), (320, 320),
            #                          mode="bilinear", align_corners=True).squeeze(1).long()
            # ndsmVal = torch.repeat_interleave(ndsmVal, 3, dim=1)
            # teacherVal, studentVal = net(imageVal, ndsmVal)
            # outVal = net(imageVal)
            outVal = net(imageVal, ndsmVal)
            loss = criterion_without(outVal[0:4], labelVal)
            outVal = outVal[0].max(dim=1)[1].data.cpu().numpy()
            # outVal = outVal[1].max(dim=1)[1].data.cpu().numpy()
            labelVal = labelVal.data.cpu().numpy()
            accval = accuary(outVal, labelVal)
            # print('accVal:', accval)
            print('Valid:    Iteration {:4}'.format(j), 'Loss:', loss.item())
            eval_loss = loss.item() + eval_loss
            acc = acc + accval

    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prec_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    epoch_str = ('Epoch: {}, Train Loss: {:.5f},Valid Loss: {:.5f},Valid Acc: {:.5f}'.format(
        epoch, train_loss / len(train_dataloader), eval_loss / len(test_dataloader), acc / len(test_dataloader)))
    time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
    print(epoch_str + time_str)

    trainlosslist_nju.append(train_loss / len(train_dataloader))
    vallosslist_nju.append(eval_loss / len(test_dataloader))

    if acc / len(test_dataloader) >= max(best):
        best.append(acc / len(test_dataloader))
        numloss = epoch
        # torch.save(net.state_dict(), './weight/PPNet_S_KD(CE[S,T]_KL+selfA))-{}-loss.pth'.format(DATASET))
        torch.save(net.state_dict(), './weight/kd/fusionc/1109p-{}-loss.pth'.format(DATASET))



    if epoch > 4 :
        torch.save(net.state_dict(), './weight/kd/fusionc/1109p-{}-{}-{}-loss.pth'.format(epoch, DATASET,acc / len(test_dataloader)))



    print(max(best), '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   Accuracy', numloss)

# loss4          loss =  loss_label0 + loss3  frequency_kd2
# loss4_1    loss =  loss_label0 + loss3  frequency_kd3
# loss = loss_label0 + loss1 + loss2 + loss4  loss3
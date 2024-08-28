import torch
import torch.nn as nn
from torch.nn import functional as F

from toolbox.backbone.mobilenet.MobileNetv2 import mobilenet_v2

from toolbox.model1.paper1_12.student7_3.student7_3_3.encoder_fusion1 import fusionmoduleteacher1low,fusionmoduleteacher0high

from toolbox.model1.paper1_12.student7_3.student7_3_3.decoder import PyramidPooladd0, SAM0, DecoderBlock0

"""the same use as teacher7_5reduce , 不能//2
    """
###############################################################################

class student1(nn.Module):
    def __init__(self,  channels=[16, 24, 32, 160, 320]):
        super(student1, self).__init__()
        self.channels = channels

        mobilenet_raw_model1 = mobilenet_v2(pretrained=True)
        mobilenet_raw_model2 = mobilenet_v2(pretrained=True)
        ###############################################
        #Backbone model
        self.encoder_thermal_layer0 = mobilenet_raw_model1.features[0:2]
        self.encoder_thermal_layer1 = mobilenet_raw_model1.features[2:4]
        self.encoder_thermal_layer2 = mobilenet_raw_model1.features[4:7]
        self.encoder_thermal_layer3 = mobilenet_raw_model1.features[7:17]
        self.encoder_thermal_layer4 = mobilenet_raw_model1.features[17:18]

        self.encoder_rgb_layer0 = mobilenet_raw_model2.features[0:2]
        self.encoder_rgb_layer1 = mobilenet_raw_model2.features[2:4]
        self.encoder_rgb_layer2 = mobilenet_raw_model2.features[4:7]
        self.encoder_rgb_layer3 = mobilenet_raw_model2.features[7:17]
        self.encoder_rgb_layer4 = mobilenet_raw_model2.features[17:18]

        ###############################################
        # funsion encoders #

        self.fu_1 = fusionmoduleteacher1low(self.channels[1], self.channels[1], self.channels[1]//2, self.channels[1])

        
        self.fu_2 = fusionmoduleteacher1low(self.channels[2], self.channels[2], self.channels[2]//2, self.channels[2])

        
        self.fu_3 = fusionmoduleteacher0high(self.channels[3], self.channels[3], self.channels[3]//2, self.channels[3])


        self.fu_4 = fusionmoduleteacher0high(self.channels[4], self.channels[4], self.channels[4]//2, self.channels[4])


        ###############################################
        # decoders #
        ###############################################
        # enhance receive field #
        self.PCIM4 = PyramidPooladd0(self.channels[4], self.channels[4])
        self.PCIM3 = PyramidPooladd0(self.channels[3], self.channels[3])
        self.PCIM2 = PyramidPooladd0(self.channels[2], self.channels[2])
        self.PCIM1 = PyramidPooladd0(self.channels[1], self.channels[1])

        self.SAM4 = SAM0(self.channels[4], self.channels[4])
        self.SAM3 = SAM0(self.channels[3]*2, self.channels[3]*2)
        self.SAM2 = SAM0(self.channels[2]*2, self.channels[2]*2)
        self.SAM1 = SAM0(self.channels[1]*2, self.channels[1]*2)

        self.decoder4 = DecoderBlock0(self.channels[4], self.channels[3])
        self.decoder3 = DecoderBlock0(self.channels[3]*2, self.channels[2])
        self.decoder2 = DecoderBlock0(self.channels[2]*2, self.channels[1])


        self.ful_conv_out = nn.Sequential(
            nn.Conv2d(48, 6, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
                                          )

        self.out4 = nn.Sequential(
            nn.Conv2d(320, 6, 1, 1, 0),
            nn.Upsample(scale_factor=32,mode='bilinear', align_corners=True)
        )

        self.out3 = nn.Sequential(
            nn.Conv2d(320, 6, 1, 1, 0),
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        )

        self.out2 = nn.Sequential(
            nn.Conv2d(64, 6, 1, 1, 0),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        )



                

    def forward(self, rgb, d):
        ###############################################
        # Backbone model

        rgb0 = self.encoder_rgb_layer0(rgb)

        d0 = self.encoder_thermal_layer0(d)
        ####################################################
        ## fusion
        ####################################################
        ## layer0 直接相加


        rgb1 = self.encoder_rgb_layer1(rgb0)

        d1 = self.encoder_thermal_layer1(d0)

        ## layer1 融合
        f1 = self.fu_1(rgb1, d1)
        ## 传输到encoder2



        rgb2 = self.encoder_rgb_layer2(rgb1)
        d2 = self.encoder_thermal_layer2(d1)

        ## layer2 融合
        f2 = self.fu_2(rgb2, d2)
        ## 传输到encoder3


        rgb3 = self.encoder_rgb_layer3(rgb2)
        d3 = self.encoder_thermal_layer3(d2)

        ## layer3 融合
        f3 = self.fu_3(rgb3, d3)
        ## 传输到encoder4


        rgb4 = self.encoder_rgb_layer4(rgb3)
        d4 = self.encoder_thermal_layer4(d3)

        ## layer4 融合
        f4 = self.fu_4(rgb4, d4)


        ####################################################
        ## decoder
        ####################################################
        ## enhance

        f4 = self.PCIM4(f4)
        f3 = self.PCIM3(f3)
        f2 = self.PCIM2(f2)
        f1 = self.PCIM1(f1)

        ## decoder4
        sam4 = self.SAM4(f4)


        deco_4 =  torch.cat([self.decoder4(sam4), f3], dim=1)



        ## decoder3
        sam3 = self.SAM3(deco_4)

        deco_3 = torch.cat([self.decoder3(sam3), f2], dim=1)


        ## decoder2
        sam2 = self.SAM2(deco_3)

        deco_2 = torch.cat([self.decoder2(sam2), f1], dim=1)

        out =  self.ful_conv_out(deco_2)
        out2 = self.out2(sam2)
        out3 = self.out3(sam3)
        out4 = self.out4(sam4)

        return out, out2, out3, out4,  f1, f2, f3, f4, rgb0, d0
        # return out, out2, out3, out4



if __name__ == '__main__':
    rgb = torch.randn(4, 3, 256, 256)
    d = torch.randn(4, 3, 256, 256)

    model = student1()

    a = model(rgb, d)

    print(a[0].shape)
    print(a[1].shape)
    print(a[2].shape)
    print(a[3].shape)
    print(a[4].shape)
    print(a[5].shape)
    print(a[6].shape)
    print(a[7].shape)
    print(a[8].shape)
    print(a[9].shape)

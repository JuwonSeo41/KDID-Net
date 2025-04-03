import torch
import torch.nn as nn
from pretrainedmodels import inceptionresnetv2
import torch.nn.functional as F


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001, # value found in tensorflow
                                 momentum=0.1, # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_5b(nn.Module):

    def __init__(self):
        super(Mixed_5b, self).__init__()

        self.branch0 = BasicConv2d(192, 96, kernel_size=1, stride=1)

        self.branch1 = BasicConv2d(192, 64, kernel_size=5, stride=1, padding=2)

        self.branch2 = BasicConv2d(192, 96, kernel_size=3, stride=1, padding=1)

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(192, 64, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block35(nn.Module):

    def __init__(self, scale=1.0):
        super(Block35, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(320, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(320, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(320, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1),
            BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1)
        )

        self.conv2d = nn.Conv2d(128, 320, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_6a(nn.Module):

    def __init__(self):
        super(Mixed_6a, self).__init__()

        self.branch0 = BasicConv2d(320, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(320, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Block17(nn.Module):

    def __init__(self, scale=1.0):
        super(Block17, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(1088, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1088, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 160, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(160, 192, kernel_size=(7,1), stride=1, padding=(3,0))
        )

        self.conv2d = nn.Conv2d(384, 1088, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_7a(nn.Module):

    def __init__(self):
        super(Mixed_7a, self).__init__()

        self.branch0 = BasicConv2d(1088, 384, kernel_size=3, stride=2)

        self.branch1 = BasicConv2d(1088, 288, kernel_size=3, stride=2)

        self.branch2 = nn.Sequential(
            BasicConv2d(1088, 288, kernel_size=1, stride=1),
            BasicConv2d(288, 320, kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class FPN(nn.Module):

    def __init__(self, norm_layer, num_filters=256):

        super().__init__()
        # self.inception = inceptionresnetv2(num_classes=1000, pretrained='imagenet')

        # start at (4, 3, 128, 128)

        # self.enc0 = self.inception.conv2d_1a
        self.enc0 = BasicConv2d(3, 32, kernel_size=3, stride=2)     # (4, 32, 63, 63)

        # self.enc1 = nn.Sequential(
        #     self.inception.conv2d_2a,   # (4, 32, 61, 61)
        #     self.inception.conv2d_2b,   # (4, 64, 61, 61)
        #     self.inception.maxpool_3a,  # (4, 64, 30, 30)
        # ) # 64
        self.enc1 = nn.Sequential(
            BasicConv2d(32, 64, kernel_size=3, stride=1),    # (4, 64, 61, 61)
            nn.MaxPool2d(3, stride=2)   # (4, 64, 30, 30)
        )
        # self.enc2 = nn.Sequential(
        #     self.inception.conv2d_3b,   # (4, 80, 30, 30)
        #     self.inception.conv2d_4a,   # (4, 192, 28, 28)
        #     self.inception.maxpool_5a,  # (4, 192, 13, 13)
        # )  # 192
        self.enc2 = nn.Sequential(
            BasicConv2d(64, 192, kernel_size=3, stride=1),   # (4, 192, 28, 28)
            nn.MaxPool2d(3, stride=2)   # (4, 192, 13, 13)
        )
        # self.enc3 = nn.Sequential(
        #     self.inception.mixed_5b,
        #     self.inception.repeat,
        #     self.inception.mixed_6a,
        # )   # 1088
        self.enc3 = nn.Sequential(
            Mixed_5b(),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Mixed_6a()                  # (4, 1088, 6, 6)
        )
        # self.enc4 = nn.Sequential(
        #     self.inception.repeat_1,
        #     self.inception.mixed_7a,
        # ) #2080
        self.enc4 = nn.Sequential(
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Mixed_7a()                  # (4, 2080, 2, 2)
        )

        self.td1 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU())
        self.td2 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU())
        self.td3 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU())
        self.pad = nn.ReflectionPad2d(1)
        self.lateral4 = nn.Conv2d(2080, num_filters, kernel_size=1, bias=False)
        self.lateral3 = nn.Conv2d(1088, num_filters, kernel_size=1, bias=False)
        self.lateral2 = nn.Conv2d(192, num_filters, kernel_size=1, bias=False)
        self.lateral1 = nn.Conv2d(64, num_filters, kernel_size=1, bias=False)
        self.lateral0 = nn.Conv2d(32, num_filters // 2, kernel_size=1, bias=False)

        # for param in self.inception.parameters():
        #     param.requires_grad = False

    # def unfreeze(self):
        # for param in self.inception.parameters():
        #     param.requires_grad = True

    def forward(self, x):

        # Bottom-up pathway, from ResNet
        # x = (4, 3, 128, 128) -> (B, C, W, H)
        enc0 = self.enc0(x) # enc0 = (4, 32, 63, 63)

        enc1 = self.enc1(enc0) # 256 enc1 = (4, 64, 30, 30)

        enc2 = self.enc2(enc1) # 512 enc2 = (4, 192, 13, 13)

        enc3 = self.enc3(enc2) # 1024 enc3 = (4, 1088, 6, 6)

        enc4 = self.enc4(enc3) # 2048 enc4 = (4, 2080, 2, 2)

        # Lateral connections
        lateral4 = self.pad(self.lateral4(enc4))    # (4, 256, 4, 4)

        lateral3 = self.pad(self.lateral3(enc3))    # (4, 256, 8, 8)

        lateral2 = self.lateral2(enc2)              # (4, 256, 13, 13)

        lateral1 = self.pad(self.lateral1(enc1))    # (4, 256, 32, 32)

        lateral0 = self.lateral0(enc0)              # (4, 128, 63, 63)

        # Top-down pathway
        pad = (1, 2, 1, 2)  # pad last dim by 1 on each side
        pad1 = (0, 1, 0, 1)
        map4 = lateral4
        map3 = self.td1(lateral3 + nn.functional.upsample(map4, scale_factor=2, mode="nearest"))    # (4, 256, 8, 8)
        map2 = self.td2(F.pad(lateral2, pad, "reflect") + nn.functional.upsample(map3, scale_factor=2, mode="nearest")) # (4, 256, 16, 16)
        map1 = self.td3(lateral1 + nn.functional.upsample(map2, scale_factor=2, mode="nearest"))    # (4, 256, 32, 32)
        return enc0, enc2, F.pad(lateral0, pad1, "reflect"), map1, map2, map3, map4   # F.pad() = map0 = (4, 128, 32, 32)


class FPNHead(nn.Module):
    def __init__(self, num_in, num_out):
        super().__init__()

        self.block0 = nn.Conv2d(num_in, num_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = nn.functional.relu(self.block0(x))
        return x


class FPNInception(nn.Module):

    def __init__(self, norm_layer, output_ch=3, num_filters=128, num_filters_fpn=256):
        super().__init__()

        # Feature Pyramid Network (FPN) with four feature maps of resolutions
        # 1/4, 1/8, 1/16, 1/32 and `num_filters` filters for all feature maps.
        self.fpn = FPN(num_filters=num_filters_fpn, norm_layer=norm_layer)

        # The segmentation heads on top of the FPN

        self.head1 = FPNHead(num_filters_fpn, num_filters)
        self.head2 = FPNHead(num_filters_fpn, num_filters)
        self.head3 = FPNHead(num_filters_fpn, num_filters)
        self.head4 = FPNHead(num_filters_fpn, num_filters)

        self.smooth = nn.Sequential(
            nn.Conv2d(4 * num_filters, num_filters, kernel_size=3, padding=1),
            norm_layer(num_filters),
            nn.ReLU(),
        )

        self.smooth2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters // 2, kernel_size=3, padding=1),
            norm_layer(num_filters // 2),
            nn.ReLU(),
        )

        self.final = nn.Conv2d(num_filters // 2, output_ch, kernel_size=3, padding=1)

        # Classification head 추가
        # num_classes = 38
        # self.classifier = nn.Sequential(
        #     nn.Conv2d(2080, 512, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(512, num_classes, kernel_size=1)
        # )
        # self.GAP = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        with torch.autograd.set_detect_anomaly(True):
            enc0, enc2, map0, map1, map2, map3, map4 = self.fpn(x)

            # Classification
            # logits = self.classifier(enc4)
            # logits = self.GAP(logits)
            # logits = logits.view(logits.size(0), -1)  # Flatten
                                                                                                # map0 = (4, 128, 64, 64)
            map4 = nn.functional.upsample(self.head4(map4), scale_factor=8, mode="nearest")     # (4, 256, 4, 4) -> (4. 128, 32, 32)
            map3 = nn.functional.upsample(self.head3(map3), scale_factor=4, mode="nearest")     # (4, 256, 8, 8) -> (4, 128, 32, 32)
            map2 = nn.functional.upsample(self.head2(map2), scale_factor=2, mode="nearest")     # (4, 256, 16, 16) -> (4, 128, 32, 32)
            map1 = nn.functional.upsample(self.head1(map1), scale_factor=1, mode="nearest")     # (4, 256, 32, 32) -> (4, 128, 32, 32)

            smoothed = self.smooth(torch.cat([map4, map3, map2, map1], dim=1))      # concat = (4, 128x4, 32, 32) -> (4, 128, 32, 32)
            smoothed = nn.functional.upsample(smoothed, scale_factor=2, mode="nearest")     # (4, 128, 32, 32) -> (4, 128, 64, 64)
            smoothed = self.smooth2(smoothed + map0)    # s+m0 = (4, 128, 64, 64) -> (4, 64, 64, 64)
            smoothed = nn.functional.upsample(smoothed, scale_factor=2, mode="nearest")     # (4, 64, 128, 128)

            final = self.final(smoothed)    # (4, 3, 128, 128)
            res = torch.tanh(final) + x

        return torch.clamp(res, min=-1, max=1), enc0, enc2, smoothed

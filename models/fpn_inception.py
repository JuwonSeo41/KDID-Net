import torch
import torch.nn as nn
from pretrainedmodels import inceptionresnetv2
from torchsummary import summary
import torch.nn.functional as F


# class ESA(nn.Module):
#     def __init__(self, num_feat=50, conv=nn.Conv2d):
#         super(ESA, self).__init__()
#         f = num_feat // 4
#         self.conv1 = nn.Conv2d(num_feat, f, 1)
#         self.conv_f = nn.Conv2d(f, f, 1)
#         self.maxPooling = nn.MaxPool2d(kernel_size=7, stride=3)
#         self.conv_max = conv(f, f, kernel_size=3, padding=1)
#         self.conv2 = conv(f, f, 3, 2, 0)
#         self.conv3 = conv(f, f, kernel_size=3, padding=1)
#         self.conv3_ = conv(f, f, kernel_size=3, padding=1)
#         self.conv4 = nn.Conv2d(f, num_feat, 1)
#         self.sigmoid = nn.Sigmoid()
#         self.GELU = nn.GELU()
#
#     def forward(self, input):
#         c1_ = (self.conv1(input))
#         c1 = self.conv2(c1_)
#         v_max = self.maxPooling(c1)
#         v_range = self.GELU(self.conv_max(v_max))
#         c3 = self.GELU(self.conv3(v_range))
#         c3 = self.conv3_(c3)
#         c3 = F.interpolate(c3, (input.size(2), input.size(3)), mode='bilinear', align_corners=False)
#         cf = self.conv_f(c1_)
#         c4 = self.conv4((c3 + cf))
#         m = self.sigmoid(c4)
#
#         return input * m


# class SELayer(nn.Module):
#     def __init__(self, channel):
#         super(SELayer, self).__init__()
#
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Linear(channel, channel // 16, bias=True)
#         self.fc2 = nn.Linear(channel // 16, channel, bias=True)
#         self.relu = nn.ReLU(inplace=True)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         y = self.avg_pool(x)    # 1x1xC
#         y = y.view(y.size(0), -1)
#         y = self.fc1(y)
#         y = self.relu(y)
#         y = self.fc2(y)
#         y = self.sigmoid(y)
#         y = y.view(x.size(0), x.size(1), 1, 1)      # (B, C, W, H) --> (2, 128, 1, 1)
#         return x * y


################################## for CBAM
# class BasicConv_C(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
#                  bn=True, bias=False):
#         super(BasicConv_C, self).__init__()
#         self.out_channels = out_planes
#         self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
#                               dilation=dilation, groups=groups, bias=bias)
#         self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
#         self.relu = nn.ReLU() if relu else None
#
#     def forward(self, x):
#         x = self.conv(x)
#         if self.bn is not None:
#             x = self.bn(x)
#         if self.relu is not None:
#             x = self.relu(x)
#         return x
#
#
# class Flatten(nn.Module):
#     def forward(self, x):
#         return x.view(x.size(0), -1)
#
#
# class ChannelGate(nn.Module):
#     def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
#         super(ChannelGate, self).__init__()
#         self.gate_channels = gate_channels
#         self.mlp = nn.Sequential(
#             Flatten(),
#             nn.Linear(gate_channels, gate_channels // reduction_ratio),
#             nn.ReLU(),
#             nn.Linear(gate_channels // reduction_ratio, gate_channels)
#         )
#         self.pool_types = pool_types
#
#     def forward(self, x):
#         channel_att_sum = None
#         for pool_type in self.pool_types:
#             if pool_type == 'avg':
#                 avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
#                 channel_att_raw = self.mlp(avg_pool)
#             elif pool_type == 'max':
#                 max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
#                 channel_att_raw = self.mlp(max_pool)
#             elif pool_type == 'lp':
#                 lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
#                 channel_att_raw = self.mlp(lp_pool)
#             elif pool_type == 'lse':
#                 # LSE pool only
#                 lse_pool = logsumexp_2d(x)
#                 channel_att_raw = self.mlp(lse_pool)
#
#             if channel_att_sum is None:
#                 channel_att_sum = channel_att_raw
#             else:
#                 channel_att_sum = channel_att_sum + channel_att_raw
#
#         scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
#         return x * scale
#
#
# def logsumexp_2d(tensor):
#     tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
#     s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
#     outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
#     return outputs
#
#
# class ChannelPool(nn.Module):
#     def forward(self, x):
#         return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
#
#
# class SpatialGate(nn.Module):
#     def __init__(self):
#         super(SpatialGate, self).__init__()
#         kernel_size = 7
#         self.compress = ChannelPool()
#         self.spatial = BasicConv_C(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)
#
#     def forward(self, x):
#         x_compress = self.compress(x)
#         x_out = self.spatial(x_compress)
#         scale = F.sigmoid(x_out)  # broadcasting
#         return x * scale

# class CBAM(nn.Module):
#     def __init__(self, channels):
#         super(CBAM, self).__init__()
#         self.channel_att = ChannelGate(channels)
#         self.spatial_att = SpatialGate()
#
#     def forward(self, x):
#         x = self.channel_att(x)
#         x = self.spatial_att(x)
#         return x

################################## for CBAM


# def mean_channels(F):   # for CCA
#     assert(F.dim() == 4)
#     spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
#     return spatial_sum / (F.size(2) * F.size(3))
#
#
# def stdv_channels(F):   # for CCA
#     assert(F.dim() == 4)
#     F_mean = mean_channels(F)
#     F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
#     return F_variance.pow(0.5)  # 제곱근
#
#
# class CCALayer(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(CCALayer, self).__init__()
#
#         self.contrast = stdv_channels
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv_du = nn.Sequential(
#             nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         y = self.contrast(x) + self.avg_pool(x)
#         # print("Contrast 값 : ", self.contrast(x))
#         # y = self.avg_pool(x)
#         y = self.conv_du(y)
#         return x * y


# class PAM_Module(nn.Module):
#     """ Position attention module"""
#     #Ref from SAGAN
#     def __init__(self, in_dim):
#         super(PAM_Module, self).__init__()
#         self.chanel_in = in_dim
#
#         self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x):
#         """
#             inputs :
#                 x : input feature maps( B X C X H X W)
#             returns :
#                 out : attention value + input feature
#                 attention: B X (HxW) X (HxW)
#         """
#         m_batchsize, C, height, width = x.size()        # (2, 128, 64, 64)
#         proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
#         proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
#         energy = torch.bmm(proj_query, proj_key)
#         attention = self.softmax(energy)
#         proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
#
#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))
#         out = out.view(m_batchsize, C, height, width)
#
#         out = self.gamma*out + x
#         return out


# class BasicConv_do(nn.Module):      # for FFT
#     def __init__(self, in_channel, out_channel, kernel_size, stride=1, bias=False, norm=False, relu=True,
#                  relu_method=nn.ReLU, groups=1, norm_method=nn.BatchNorm2d):
#         super(BasicConv_do, self).__init__()
#         if bias and norm:
#             bias = False
#
#         padding = kernel_size // 2
#         layers = list()
#         layers.append(nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
#         if norm:
#             layers.append(norm_method(out_channel))
#         if relu:
#             if relu_method == nn.ReLU:
#                 layers.append(nn.ReLU(inplace=True))
#             elif relu_method == nn.LeakyReLU:
#                 layers.append(nn.LeakyReLU(inplace=True))
#             else:
#                 layers.append(relu_method())
#         self.main = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.main(x)
#
# class FFT_Res(nn.Module):
#     def __init__(self, out_channel, norm='backward'):
#         super(FFT_Res, self).__init__()
#         self.main = nn.Sequential(      # 3x3conv - relu - 3x3conv
#             BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
#             BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
#         )
#         self.main_fft = nn.Sequential(  # 1x1conv - relu - 1x1conv
#             BasicConv_do(out_channel * 2, out_channel * 2, kernel_size=1, stride=1, relu=True),
#             BasicConv_do(out_channel * 2, out_channel * 2, kernel_size=1, stride=1, relu=False)
#         )
#         self.dim = out_channel
#         self.norm = norm
#
#     def forward(self, x):
#         _, _, H, W = x.shape
#         dim = 1
#         y = torch.fft.rfft2(x, norm=self.norm)
#         y_imag = y.imag     # if y = 3+4j --> y.imag = 4
#         y_real = y.real
#         y_f = torch.cat([y_real, y_imag], dim=dim)
#         y = self.main_fft(y_f)
#         y_real, y_imag = torch.chunk(y, 2, dim=dim)
#         y = torch.complex(y_real, y_imag)       # 복소수 텐서 생성, if y_real = 3 y_imag = 4 --> torch.complex() = 3+4j
#         y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
#         return self.main(x) + x + y


class FPNHead(nn.Module):
    def __init__(self, num_in, num_mid, num_out):
        super().__init__()

        self.block0 = nn.Conv2d(num_in, num_mid, kernel_size=3, padding=1, bias=False)
        self.block1 = nn.Conv2d(num_mid, num_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = nn.functional.relu(self.block0(x))
        x = nn.functional.relu(self.block1(x))
        return x

class ConvBlock(nn.Module):
    def __init__(self, num_in, num_out, norm_layer):
        super().__init__()

        self.block = nn.Sequential(nn.Conv2d(num_in, num_out, kernel_size=3, padding=1),
                                 norm_layer(num_out),
                                 nn.ReLU())

    def forward(self, x):
        x = self.block(x)
        return x


class FPNInception(nn.Module):

    def __init__(self, norm_layer, output_ch=3, num_filters=128, num_filters_fpn=256):
        super().__init__()

        # Feature Pyramid Network (FPN) with four feature maps of resolutions
        # 1/4, 1/8, 1/16, 1/32 and `num_filters` filters for all feature maps.
        self.fpn = FPN(num_filters=num_filters_fpn, norm_layer=norm_layer)

        # The segmentation heads on top of the FPN

        self.head1 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head2 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head3 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head4 = FPNHead(num_filters_fpn, num_filters, num_filters)

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
        num_classes = 38
        self.classifier = nn.Sequential(
            nn.Conv2d(2080, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )
        self.GAP = nn.AdaptiveAvgPool2d(1)

    def unfreeze(self):
        self.fpn.unfreeze()

    def forward(self, x):
        with torch.autograd.set_detect_anomaly(True):
            enc0, enc2, enc4, map0, map1, map2, map3, map4 = self.fpn(x)

         # Classification
            logits = self.classifier(enc4)
            logits = self.GAP(logits)
            logits = logits.view(logits.size(0), -1)  # Flatten
                                                                                            # map0 = (4, 128, 64, 64)
            map4 = nn.functional.upsample(self.head4(map4), scale_factor=8, mode="nearest") # (4, 256, 4, 4) -> (4. 128, 32, 32)
            map3 = nn.functional.upsample(self.head3(map3), scale_factor=4, mode="nearest") # (4, 256, 8, 8) -> (4, 128, 32, 32)
            map2 = nn.functional.upsample(self.head2(map2), scale_factor=2, mode="nearest") # (4, 256, 16, 16) -> (4, 128, 32, 32)
            map1 = nn.functional.upsample(self.head1(map1), scale_factor=1, mode="nearest") # (4, 256, 32, 32) -> (4, 128, 32, 32)

            smoothed = self.smooth(torch.cat([map4, map3, map2, map1], dim=1))  # concat = (4, 128x4, 32, 32) -> (4, 128, 32, 32)
            smoothed = nn.functional.upsample(smoothed, scale_factor=2, mode="nearest") # (4, 128, 32, 32) -> (4, 128, 64, 64)
            smoothed = self.smooth2(smoothed + map0)    # + = (4, 128, 64, 64) -> (4, 64, 64, 64)
            smoothed = nn.functional.upsample(smoothed, scale_factor=2, mode="nearest") # (4, 64, 128, 128)

            final = self.final(smoothed)    # (4, 3, 128, 128)
            res = torch.tanh(final) + x

        return torch.clamp(res, min = -1,max = 1), logits, enc0, enc2, smoothed


class FPN(nn.Module):

    def __init__(self, norm_layer, num_filters=256):
        """Creates an `FPN` instance for feature extraction.
        Args:
          num_filters: the number of filters in each output pyramid level
          pretrained: use ImageNet pre-trained backbone feature extractor
        """

        super().__init__()
        self.inception = inceptionresnetv2(num_classes=1000, pretrained='imagenet')

        self.enc0 = self.inception.conv2d_1a ##in = 3, out = 32
        self.enc1 = nn.Sequential(
            self.inception.conv2d_2a,## in 32 out 64
            self.inception.conv2d_2b,
            self.inception.maxpool_3a,
        ) # 64
        self.enc2 = nn.Sequential(
            self.inception.conv2d_3b,
            self.inception.conv2d_4a,
            self.inception.maxpool_5a,
        )  # 192
        self.enc3 = nn.Sequential(
            self.inception.mixed_5b,
            self.inception.repeat,
            self.inception.mixed_6a,
        )   # 1088
        self.enc4 = nn.Sequential(
            self.inception.repeat_1,
            self.inception.mixed_7a,
        ) #2080

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

        for param in self.inception.parameters():
            param.requires_grad = False

        ########## attention module ##########
        # 정의된 부분이랑 여기, 그리고 forward 부분 3개 고칠 것
        # self.ESA = ESA(256)
        # self.ESA_ = ESA(128)
        # self.SE = SELayer(256)
        # self.SE_ = SELayer(128)
        # self.CBAM = CBAM(256)
        # self.CBAM_ = CBAM(128)
        # self.FFT_res = FFT_Res(256)
        # self.FFT_res_ = FFT_Res(128)
        # self.CCA = CCALayer(256)
        # self.CCA_ = CCALayer(128)
        # self.PAM = PAM_Module(256)
        # self.PAM_ = PAM_Module(128)

    def unfreeze(self):
        for param in self.inception.parameters():
            param.requires_grad = True

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
        # lateral4 = self.SE(lateral4) + lateral4  # SE
        # lateral4 = self.CCA(lateral4) + lateral4    # CCA
        # lateral4 = self.PAM(lateral4)  # PAM

        lateral3 = self.pad(self.lateral3(enc3))    # (4, 256, 8, 8)
        # lateral3 = self.SE(lateral3) + lateral3  # SE
        # lateral3 = self.FFT_res(lateral3)   # FFT Res
        # lateral3 = self.CCA(lateral3) + lateral3    # CCA
        # lateral3 = self.PAM(lateral3)  # PAM

        lateral2 = self.lateral2(enc2)              # (4, 256, 13, 13)
        # lateral2 = self.SE(lateral2) + lateral2  # SE
        # lateral2 = self.CBAM(lateral2) + lateral2   # CBAM
        # lateral2 = self.FFT_res(lateral2)       # FFT Res
        # lateral2 = self.CCA(lateral2) + lateral2    # CCA
        # lateral2 = self.PAM(lateral2)  # PAM

        lateral1 = self.pad(self.lateral1(enc1))    # (4, 256, 32, 32)
        # lateral1 = self.ESA(lateral1) + lateral1    # ESA
        # lateral1 = self.SE(lateral1) + lateral1  # SE
        # lateral1 = self.CBAM(lateral1) + lateral1   # CBAM
        # lateral1 = self.FFT_res(lateral1)       # FFT Res
        # lateral1 = self.CCA(lateral1) + lateral1    # CCA
        # lateral1 = self.PAM(lateral1)  # PAM

        lateral0 = self.lateral0(enc0)              # (4, 128, 63, 63)
        # lateral0 = self.ESA_(lateral0) + lateral0   # ESA
        # lateral0 = self.SE_(lateral0) + lateral0  # SE
        # lateral0 = self.CBAM_(lateral0) + lateral0  # CBAM
        # lateral0 = self.FFT_res_(lateral0)      # FFT Res
        # lateral0 = self.CCA_(lateral0) + lateral0   # CCA
        # lateral0 = self.PAM_(lateral0)  # PAM

        # Top-down pathway
        pad = (1, 2, 1, 2)  # pad last dim by 1 on each side
        pad1 = (0, 1, 0, 1)
        map4 = lateral4
        map3 = self.td1(lateral3 + nn.functional.upsample(map4, scale_factor=2, mode="nearest"))    # (4, 256, 8, 8)
        map2 = self.td2(F.pad(lateral2, pad, "reflect") + nn.functional.upsample(map3, scale_factor=2, mode="nearest")) # (4, 256, 16, 16)
        map1 = self.td3(lateral1 + nn.functional.upsample(map2, scale_factor=2, mode="nearest"))    # (4, 256, 32, 32)
        return enc0, enc2, enc4, F.pad(lateral0, pad1, "reflect"), map1, map2, map3, map4   # F.pad() = map0 = (4, 128, 32, 32)

import torch.nn as nn
from collections import OrderedDict
import torch
from torchsummary import torchsummary
import torch.nn.functional as F
from cbam import MS_CAM


class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, init_features=16):
        super(Unet, self).__init__()

        features = init_features
        # 原图片256 * 256 * 3 -----》256 * 256 * 16
        self.encoder1 = Unet._block(in_channels, features, name='enc1')
        # 128 * 128 * 16
        self.pool1 = nn.MaxPool2d(2, 2)
        # 128 * 128 * 32
        self.encoder2 = Unet._block(features, features * 2, name='enc2')
        # 64 * 64 * 32
        self.pool2 = nn.MaxPool2d(2, 2)
        # 64 * 64 * 64
        self.encoder3 = Unet._block(features * 2, features * 4, name='enc2')
        # 32 * 32 * 64
        self.pool3 = nn.MaxPool2d(2, 2)
        # 32 * 32 * 128
        self.encoder4 = Unet._block(features * 4, features * 8, name='enc2')
        # 16 * 16 * 128
        self.pool4 = nn.MaxPool2d(2, 2)
        # bottleneck表示瓶底的意思， 16 * 16 * 256
        self.bottleneck = Unet._block(features * 8, features * 16, name='bottleneck')

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        # 32 * 32 * 128
        self.decoder4 = Unet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        # 64 * 64 * 64
        self.decoder3 = Unet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        # 128 * 128 * 32
        self.decoder2 = Unet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        # 256 * 256 * 16
        self.decoder1 = Unet._block(features * 2, features, name="dec1")

        # 最后使用1x1的卷积核压缩通道数
        self.out = nn.Conv2d(features, out_channels, kernel_size=1)
        self.feat4_attention = MS_CAM(channel=features * 8)
        self.bottle_attention = MS_CAM(channel=features * 16)


    def forward(self, x1):
        # 编码阶段
        enc1 = self.encoder1(x1)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(self.feat4_attention(enc4)))
        bottleneck = self.bottle_attention(bottleneck)

        # 解码阶段
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        out = self.out(dec1)
        return out

    # 两个卷积块
    # OrderDict里面放的list
    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict([

                (name+'conv1', nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False)),
                (name+'norm1', nn.BatchNorm2d(features)),
                (name+'relu1', nn.ReLU(inplace=True)),


                # 注意这里输入和输出通道要相同，因为第二次卷积不包含通道数
                (name+'conv2', nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False)),
                (name+'norm2', nn.BatchNorm2d(features)),
                (name+'relu2', nn.ReLU(inplace=True)),
            ])
        )


class ChangeUnet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, init_features=16):
        super(ChangeUnet, self).__init__()

        features = init_features
        # 原图片256 * 256 * 3 -----》256 * 256 * 16
        self.encoder1 = Unet._block(in_channels, features, name='enc1')
        # 128 * 128 * 16
        self.pool1 = nn.MaxPool2d(2, 2)
        # 128 * 128 * 32
        self.encoder2 = Unet._block(features, features * 2, name='enc2')
        # 64 * 64 * 32
        self.pool2 = nn.MaxPool2d(2, 2)
        # 64 * 64 * 64
        self.encoder3 = Unet._block(features * 2, features * 4, name='enc2')
        # 32 * 32 * 64
        self.pool3 = nn.MaxPool2d(2, 2)
        # 32 * 32 * 128
        self.encoder4 = Unet._block(features * 4, features * 8, name='enc2')
        # 16 * 16 * 128
        self.pool4 = nn.MaxPool2d(2, 2)
        # bottleneck表示瓶底的意思， 16 * 16 * 256
        self.bottleneck = Unet._block(features * 8, features * 16, name='bottleneck')

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        # 32 * 32 * 128
        self.decoder4 = Unet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        # 64 * 64 * 64
        self.decoder3 = Unet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        # 128 * 128 * 32
        self.decoder2 = Unet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        # 256 * 256 * 16
        self.decoder1 = Unet._block(features * 2, features, name="dec1")

        # 最后使用1x1的卷积核压缩通道数
        self.out = nn.Conv2d(features * 2, out_channels, kernel_size=1)

    def forward(self, x1, x2):
        # 编码阶段 x1
        enc1_1 = self.encoder1(x1)
        enc2_1 = self.encoder2(self.pool1(enc1_1))
        enc3_1 = self.encoder3(self.pool2(enc2_1))
        enc4_1 = self.encoder4(self.pool3(enc3_1))
        bottleneck = self.bottleneck(self.pool4(enc4_1))

        # 解码阶段 x1
        dec4_1 = self.upconv4(bottleneck)
        dec4_1 = torch.cat((dec4_1, enc4_1), dim=1)
        dec4_1 = self.decoder4(dec4_1)

        dec3_1 = self.upconv3(dec4_1)
        dec3_1 = torch.cat((dec3_1, enc3_1), dim=1)
        dec3_1 = self.decoder3(dec3_1)

        dec2_1 = self.upconv2(dec3_1)
        dec2_1 = torch.cat((dec2_1, enc2_1), dim=1)
        dec2_1 = self.decoder2(dec2_1)

        dec1_1 = self.upconv1(dec2_1)
        dec1_1 = torch.cat((dec1_1, enc1_1), dim=1)
        dec1_1 = self.decoder1(dec1_1)


        # 编码阶段 x2
        enc1_2 = self.encoder1(x2)
        enc2_2 = self.encoder2(self.pool1(enc1_2))
        enc3_2 = self.encoder3(self.pool2(enc2_2))
        enc4_2 = self.encoder4(self.pool3(enc3_2))
        bottleneck = self.bottleneck(self.pool4(enc4_2))

        # 解码阶段 x2
        dec4_2 = self.upconv4(bottleneck)
        dec4_2 = torch.cat((dec4_2, enc4_2), dim=1)
        dec4_2 = self.decoder4(dec4_2)

        dec3_2 = self.upconv3(dec4_2)
        dec3_2 = torch.cat((dec3_2, enc3_2), dim=1)
        dec3_2 = self.decoder3(dec3_2)

        dec2_2 = self.upconv2(dec3_2)
        dec2_2 = torch.cat((dec2_2, enc2_2), dim=1)
        dec2_2 = self.decoder2(dec2_2)

        dec1_2 = self.upconv1(dec2_2)
        dec1_2 = torch.cat((dec1_2, enc1_2), dim=1)
        dec1_2 = self.decoder1(dec1_2)


        # 将特征的通道进行融合
        output = torch.cat((dec1_1, dec1_2), dim=1)
        return self.out(output)

    # 两个卷积块
    # OrderDict里面放的list
    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict([

                (name+'conv1', nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False)),
                (name+'norm1', nn.BatchNorm2d(features)),
                (name+'relu1', nn.ReLU(inplace=True)),


                # 注意这里输入和输出通道要相同，因为第二次卷积不包含通道数
                (name+'conv2', nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False)),
                (name+'norm2', nn.BatchNorm2d(features)),
                (name+'relu2', nn.ReLU(inplace=True)),
            ])
        )


class PSPModule(nn.Module):
    def __init__(self, in_channels, pool_sizes, norm_layer):  # pool_sizes 可以有 6 3 2 1 等组合，即池化的大小
        super(PSPModule, self).__init__()
        out_channels = in_channels // len(pool_sizes)
        self.stages = nn.ModuleList(
            [self._make_stages(in_channels, out_channels, pool_size, norm_layer) for pool_size in pool_sizes ]
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+(out_channels * len(pool_sizes)), out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )  # 将堆叠的特征层进行通道数的调整
    def _make_stages(self, in_channels, out_channels, bin_sz, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)  # 可以指定任意大小的输出形状
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)  # 首先利用1x1的卷积进行通道数的调整
        bn = norm_layer(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output



if __name__ == '__main__':
    i1 = torch.randn(1, 3, 256, 256)
    i2 = torch.randn(1, 3, 256, 256)
    model = Unet(init_features=64)
    out = model(i1)
    # output = model(i1, i2)
    print('sss')
    # model = ChangeUnet()
    torchsummary.summary(model, input_size=(3, 256, 256))
    image = torch.randn(32, 3, 256, 256)
    # ppm = PSPModule(3, [1, 2, 3, 6], nn.BatchNorm2d())

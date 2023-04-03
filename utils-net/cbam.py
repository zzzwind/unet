import torch
from torch import nn
from ecanet import ECANet


class ChannelAttention(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        max_pool_out = self.max_pool(x).view([b, c])
        avg_pool_out = self.avg_pool(x).view([b, c])

        max_fc_out = self.fc(max_pool_out)
        avg_fc_out = self.fc(avg_pool_out)

        out = max_fc_out + avg_fc_out
        out = self.sigmoid(out).view([b, c, 1, 1])
        return out * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding='same')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        max_pool_out, _ = torch.max(x, dim=1, keepdim=True)  # 需要将通道这一维度保留下来
        avg_pool_out = torch.mean(x, dim=1, keepdim=True)
        pool_out = torch.cat([max_pool_out, avg_pool_out], dim=1)
        out = self.conv(pool_out)
        out = self.sigmoid(out)
        return out * x




class CBAM(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channel, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class MS_CAM(nn.Module):
    def __init__(self, channel, kernel_size=7):
        super(MS_CAM, self).__init__()
        self.channel_attention = ECANet(channel=channel)
        self.spatial_attention = SpatialAttention(kernel_size)
        self.conv = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, padding=1, bias=False)

    def forward(self, x):

        # 通道注意力机制这一个分枝
        fca = self.channel_attention(x)
        fca = fca + x
        out1 = self.conv(fca)

        # 空间注意力机制这个分枝
        fsa = self.spatial_attention(x)
        out2 = out1 + fsa

        out = out1 + out2 + x
        return out

        pass


if __name__ == '__main__':
    # image = torch.randn(2, 512, 256, 256)
    # cbam = CBAM(512)
    # out = cbam(image)
    # import pydensecrf.densecrf as dcrf
    #
    # print(cbam)
    image = torch.randn(2, 512, 512, 512)
    bam = MS_CAM(channel=512)
    out = bam(image)
    print(out)
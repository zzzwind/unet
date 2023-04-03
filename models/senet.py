import torch
from torch import nn

# 通道注意力的一个实现
class SENet(nn.Module):
    def __init__(self, channel, ratio = 16):
        super(SENet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // 16, False),
            nn.ReLU(),
            nn.Linear(channel // 16, channel, False),
            nn.Sigmoid()
        )

    def forward(self, x ):
        b, c, h, w = x.size()

        # b, c, 1, 1
        avg = self.avg_pool(x).view([b, c])  # 去掉最后两个1，1
        fc = self.fc(avg).view([b, c, 1, 1])
        return x * fc  # 得到了权值之后再乘以原来的变量


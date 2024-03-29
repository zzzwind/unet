import torch
from torch import nn
import math

class ECANet(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECANet, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        avg = self.avg_pool(x).view([b, 1, c])
        out = self.conv(avg)
        out = self.sigmoid(out).view([b, c, 1, 1])
        return out * x

if __name__ == "__main__":
    a = ECANet(channel=512)
    image = torch.randn(2,3,256, 256)
    out = a(image)
    print(out)
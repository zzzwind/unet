import torch.nn as nn
import torchvision
import torch
from torchvision.models.feature_extraction import create_feature_extractor

# 上采样模块（反卷积）
class Deconvolution(nn.Module):
    def __init__(self, input_channels, num_classes, input_size):
        super(Deconvolution, self).__init__()
        self.FC = nn.Sequential(
            nn.BatchNorm2d(input_channels),
            nn.Conv2d(input_channels, num_classes, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_classes)
        )
        # 如果效果不好，可以更改成github上，先反卷积再上采样
        self.Deconv = nn.UpsamplingBilinear2d(input_size)

    def forward(self, x1, x2):
        x1 = self.FC(x1)
        x1 = self.Deconv(x1)
        x2 = self.FC(x2)
        x2 = self.Deconv(x2)
        return torch.cat([x1, x2], dim=1)


class ChangeNet(nn.Module):
    def __init__(self, input_size=(256, 256), num_classes=2):
        super(ChangeNet, self).__init__()
        self.backbone = torchvision.models.vgg16(weights=None).features
        self.backbone = create_feature_extractor(
            self.backbone,
            return_nodes= {
                '16': 'feat1',
                '23': 'feat2',
                '30': 'feat3'
            }
        )

        self.cp3_Deconv = Deconvolution(256, num_classes, input_size)
        self.cp4_Deconv = Deconvolution(512, num_classes, input_size)
        self.cp5_Deconv = Deconvolution(512, num_classes, input_size)

        self.FC = nn.Conv2d(2 * num_classes, num_classes , kernel_size=1)

    def forward(self, pre, post):
        x1 = self.backbone(pre)
        x2 = self.backbone(post)
        pre1, pre2, pre3 = x1['feat1'], x1['feat2'], x1['feat3']
        post1, post2, post3 = x1['feat1'], x1['feat2'], x1['feat3']

        output1 = self.FC(self.cp3_Deconv(pre1, post1))
        output2 = self.FC(self.cp4_Deconv(pre2, post2))
        output3 = self.FC(self.cp5_Deconv(pre3, post3))

        output = output1.add(output2)
        output = output.add(output3)
        return output

if __name__ == '__main__':
    # print(torchvision.models.vgg16().features)
    # m = torchvision.models.vgg16().features
    # m = create_feature_extractor(m, return_nodes= {
    #             '9': 'feat1',
    #             '16': 'feat2',
    #             '23': 'feat3'
    #         })
    # out = m(torch.randn(1,3, 256, 256))
    # print([(k, v.shape) for k, v in out.items()])
    pre = torch.randn(1, 3, 256, 256)
    post = torch.randn(1, 3, 256, 256)
    net = ChangeNet()
    ouput = net(pre, post)
    print('ss')


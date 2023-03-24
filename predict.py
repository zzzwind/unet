import os

import numpy as np
import torch
import torch.nn as nn
from models.unet import ChangeUnet
from models.changenet import ChangeNet
from data.dataset import ChangeNetDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from metrics import *
from PIL import Image
from torchvision import transforms
import cv2
from utils.image_tools import *

transform = transforms.Compose([
    transforms.ToTensor()
])


def batch():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ChangeUnet(in_channels=3, out_channels=2)
    model.load_state_dict(torch.load('param/change-net/change-net-epoch-190-199.pth'))

    model.to(device)

    data_path = r'F:\dataset\LEVIR-CD\changed-path'
    dataset = ChangeNetDataset(path=data_path)

    test_dl = DataLoader(
        dataset=dataset,
        batch_size=32,
        # num_workers=os.cpu_count()
    )

    model.eval()

    pbar = tqdm(enumerate(test_dl), total=len(test_dl), desc="测试集")

    with torch.no_grad():
        total_acc = 0.0
        total_f1_score = 0.0
        total_miou = 0.0
        for data in pbar:
            step, (before_image, after_image, label) = data
            before_image, after_image, label = before_image.to(device), after_image.to(device), label.to(device)
            output = model(before_image, after_image)
            pred = torch.argmax(output, dim=1)

            acc = get_pixel_accuracy(pred=output, label=label, num_classes=2)
            total_acc += acc.item()
            f1 = get_f1_score(pred=output, label=label, num_classes=2)
            total_f1_score += f1.item()
            miou = get_miou(pred=output, label=label, num_classes=2)
            total_miou += miou.item()
            pbar.set_postfix(acc={format(acc.item(), '.3f')}, f1={format(f1.item(), '.3f')},
                             miou={format(miou.item(), '.3f')})

        print("acc: {:.3f}".format(total_acc / len(test_dl)), "f1: {:.3f}".format(total_f1_score / len(test_dl)),
              "miou: {:.3f}".format(total_miou / len(test_dl)))


def predict_image():

    model = ChangeNet()
    model.load_state_dict(torch.load('param/Real-change-net/latest_changenet.pth'))
    model.cuda()
    _input1 = input('please input before image path:')
    _input2 = input('please input after image path')
    # img=keep_image_size_open_rgb(_input)
    img_before_P = Image.open(_input1)
    img_after_P = Image.open(_input2)

    img_before = transform(img_before_P).cuda()  # 对于新的图片还是要进行归一化处理
    img_after = transform(img_after_P).cuda()  # 对于新的图片还是要进行归一化处理

    img_before = torch.unsqueeze(img_before, dim=0)  # 添加batch维度  因为要输入的形式必须是 b c h w
    img_after = torch.unsqueeze(img_after, dim=0)  # 添加batch维度  因为要输入的形式必须是 b c h w
    model.eval()
    out = model(img_before, img_after)
    out = torch.argmax(out, dim=1)
    out = torch.squeeze(out, dim=0)
    out = out.unsqueeze(dim=0)
    print(set((out).reshape(-1).tolist()))
    out = (out).permute((1, 2, 0)).cpu().detach().numpy()
    cv2.imwrite('result/result.png', out)
    show_image(out)
    show_image(img_before_P)
    show_image(img_after_P)


if __name__ == "__main__":
    # img = Image.open(r"F:\dataset\LEVIR-CD\val\label\val_2.png")
    # s = np.array(img)
    # print(s)
    predict_image()
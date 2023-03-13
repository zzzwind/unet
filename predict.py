import os

import torch
import torch.nn as nn
from models.unet import ChangeUnet
from data.dataset import ChangeNetDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from metrics import *

model = ChangeUnet(in_channels=3, out_channels=2)
model.load_state_dict(torch.load('weight/change-net/change-net.pth'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_path = ""
dataset = ChangeNetDataset(path=data_path)

test_dl = DataLoader(
    dataset=dataset,
    batch_size=32,
    num_workers=os.cpu_count()
)

model.eval()

pbar = tqdm(enumerate(test_dl), total=len(test_dl), desc="测试集")


with torch.no_grad:
    for data in pbar:
        step, (before_image, after_image, label) = data
        before_image, after_image, label = before_image.to(device)
        output = model(before_image, after_image)
        pred = torch.argmax(output, dim=1)

        acc = get_pixel_accuracy(pred=output, label=label, num_classes=2)
        f1 = f1_score(pred=output, label=label, num_classes=2)
        miou = miou(pred=output, label=label, num_classes=2)

        pbar.set_postfix(f"acc={format(acc.item(), '.3f')},f1={format(f1.item(), '.3f')}, miou={format(miou.item(), '.3f')},  ")







if __name__ == "__main__":
    pass
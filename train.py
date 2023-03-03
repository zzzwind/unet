from data.dataset import ChangeNetDataset
from torch.utils.data import DataLoader
from models.unet import ChangeUnet
import torch
from utils.loss_fun import *
from tqdm import tqdm
from utils.my_metircs import confusion_matrix
import os
from metrics import get_pixel_accuracy
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path = './param/change-net.pth'
img_path = r'F:\dataset\whu-change-detection\images'

full_ds = ChangeNetDataset(path=img_path)
split_rate = 0.8

train_size = int(split_rate * len(full_ds))
val_size = len(full_ds) - train_size

train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_size, val_size])
train_dl = DataLoader(
    dataset=train_ds,
    batch_size=32,
    shuffle=True,
    # num_workers=os.cpu_count(),
)
val_dl = DataLoader(
    dataset=val_ds,
    batch_size=64,
    shuffle=False,
    # num_workers=os.cpu_count(),
)

model = ChangeUnet(in_channels=3, out_channels=2, init_features=16)
model.to(device)
optimizer = torch.optim.Adam(model.parameters())
loss_fn =nn.CrossEntropyLoss()
# loss_fun = torchmetrics.Dice()

writer = SummaryWriter(log_dir='./log')

epochs = 50
for epoch in range(epochs):
    total_loss = 0
    total_acc = 0
    pbar = tqdm(enumerate(train_dl), total=len(train_dl), desc="epoch")
    model.train()
    for index, (before_image, after_image, label_image) in pbar:
        before_image, after_image, label_image = before_image.to(device), after_image.to(device), label_image.to(device)
        output = model(before_image, after_image)
        # loss = 1 - loss_fun(output, label_image.int())
        # loss.requires_grad_(True)
        loss = loss_fn(output, label_image.long())
        total_loss += loss.item() * output.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = get_pixel_accuracy(pred=output, label=label_image, num_classes=2)
        total_acc += accuracy.item() * output.size(0)
        pbar.set_description(f"Epoch [{epoch+1}/{epochs}]")
        pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item())
        writer.add_scalar('Loss/train', loss, epoch)
        writer.add_scalar('Accuracy/train', accuracy, epoch)

        single_output = output.cuda().data.cpu()
        single_output = single_output[0]

        single_label = label_image.cuda().data.cpu()
        single_label = single_label[0]
        writer.add_image('Predictions', single_output, epoch)
        writer.add_image('Labels', torch.mul(single_label, 255).unsqueeze(0), epoch)
    print(f'train_loss = {total_loss/len(train_dl)}')
    if epoch % 5 == 0:
        torch.save(model.state_dict(), weight_path)
# def train():
#


# if __name__ == '__main__':
#     print(len(train_ds))
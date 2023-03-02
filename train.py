from data.dataset import ChangeNetDataset
from torch.utils.data import DataLoader
from models.unet import ChangeUnet
import torch
from utils.loss_fun import *
from tqdm import tqdm
from utils.my_metircs import confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path = 'weight/change-net/change-net.pth'
img_path = 'images'

full_ds = ChangeNetDataset(path=img_path)
split_rate = 0.8

train_size = int(split_rate * len(full_ds))
val_size = len(full_ds) - train_size

train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_size, val_size])
train_dl = DataLoader(
    dataset=train_ds,
    batch_size=2,
    shuffle=True,
    # num_workers=4,
)
val_dl = DataLoader(
    dataset=val_ds,
    batch_size=4,
    shuffle=True,
    # num_workers=4,
)

model = ChangeUnet(in_channels=3, out_channels=2, init_features=16)

optimizer = torch.optim.Adam(model.parameters())
loss_fn =nn.CrossEntropyLoss()
# loss_fun = torchmetrics.Dice()


epochs = 50
for i in range(epochs):
    total_loss = 0
    total_acc = 0
    pbar = tqdm(enumerate(train_dl), total=len(train_dl), desc="epoch")
    model.train()
    for index, (before_image, after_image, label_image) in pbar:

        output = model(before_image, after_image)
        # loss = 1 - loss_fun(output, label_image.int())
        # loss.requires_grad_(True)
        loss = loss_fn(output, label_image.long())
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #acc
        _pred, _y = torch.max(output, dim=1)
        _t = torch.argmax(output, dim=1)
        # print(f'{i}-{i}-train_loss===>>{loss}')
        matrix = confusion_matrix(pred=_pred, label=label_image)
        pbar.set_description(f"Epoch [{i+1}/{epochs}]")
        pbar.set_postfix(loss=loss.item())
    print(f'train_loss = {total_loss/len(train_dl)}')


#
# if __name__ == '__main__':
#     print(len(train_ds))
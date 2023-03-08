from data.dataset import ChangeNetDataset
from torch.utils.data import DataLoader
from models.unet import ChangeUnet
import torch
from tqdm import tqdm
from utils.my_metircs import dice_loss
import os
from metrics import get_pixel_accuracy, f1_score, miou
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn



# 获得学习率
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_one_epoch(model, otpimizer, scheduler, loss_fn, train_loader, val_loader, device, epoch_idx):
    # 将模型设置为训练模式
    model.train()
    # 初始化损失和准确率
    train_loss = 0
    train_acc = 0
    train_f_score = 0
    train_miou = 0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch_idx}", unit="image")
    # 遍历数据集中的每个批次
    for batch in pbar:
        # 获取输入和标签，并移动到设备上
        step, (before_image, after_image, label) = batch
        before_image, after_image, label = before_image.to(device), after_image.to(device), label.to(device)

        # 前向传播，计算输出和损失(混合loss)
        outputs = model(before_image, after_image)
        batch_loss = loss_fn(outputs, label.long())
        batch_loss += dice_loss(
            F.softmax(outputs, dim=1).float(),
            F.one_hot(label.to(torch.int64), 2).permute(0, 3, 1, 2).float(),
            multiclass=True
        )
        # 后向传播，更新梯度和优化器参数
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # 累加损失和准确率
        train_loss += batch_loss.item()
        pixel_acc = get_pixel_accuracy(pred=outputs, label=label, num_classes=2)
        train_acc += pixel_acc.item()

        # 计算f-score
        _f_score = f1_score(pred=outputs, label=label, num_classes=2)
        train_f_score +=_f_score.item()

        # 计算miou
        _miou = miou(pred=outputs, label=label, num_classes=2)
        train_miou += _miou.item()

        pbar.set_postfix(**{'loss': batch_loss.item(),
                            'accuracy': pixel_acc.item(),
                            'f_score': _f_score.item(),
                            'miou': _miou.item(),
                            'lr': get_lr(optimizer)})

    # 计算平均损失和准确率，并返回结果
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    train_f_score /= len(train_loader)
    train_miou /= len(train_loader)
    # 更新学习率
    scheduler.step()
    # return loss, acc
    print(f"Epoch: {epoch_idx}, total_loss: {train_loss}, total_acc: {train_acc}, total f_score: {train_f_score}, total miou: {train_miou} lr: {get_lr(optimizer)}")
    # 验证过程
    model.eval()
    val_loss = 0
    val_acc = 0
    val_f_score = 0
    val_miou = 0
    with torch.no_grad():
        vpbar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"validation: {epoch_idx}", unit="image")
        for batch in vpbar:
            step, (before_image, after_image, label) = batch
            before_image, after_image, label = before_image.to(device), after_image.to(device), label.to(device)

            outputs = model(before_image, after_image)

            # 计算损失
            batch_loss = loss_fn(outputs, label.long())
            batch_loss += dice_loss(
                F.softmax(outputs, dim=1).float(),
                F.one_hot(label.to(torch.int64), 2).permute(0, 3, 1, 2).float(),
                multiclass=True
            )
            val_loss += batch_loss.item() * outputs

            # 计算准确率
            pixel_acc = get_pixel_accuracy(pred=outputs, label=label, num_classes=2)
            val_acc += pixel_acc.item() * outputs

            # 计算fscore
            _f_score = f1_score(pred=outputs, label=label, num_classes=2)
            val_f_score += _f_score.item() * outputs

            _miou = miou(pred=outputs, label=label, num_classes=2)
            val_miou += _miou.item() * outputs

            vpbar.set_postfix(**{'val_loss': batch_loss.item(),
                                 'val_accuracy': pixel_acc.item(),
                                 'val_f_score': _f_score.item(),
                                 'val_miou': _miou.item()
                                 })

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        val_f_score /= len(val_loader)
        val_miou /= len(val_loader)
        print(f"Epoch: {epoch_idx}, total_loss: {val_loss}, total_acc: {val_acc}, total f_score: {val_f_score} total miou：{val_miou}")




if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight_path = 'weight/change-net/change-net.pth'
    # img_path = 'images'
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
        num_workers=os.cpu_count()
    )
    val_dl = DataLoader(
        dataset=val_ds,
        batch_size=64,
        shuffle=False,
        num_workers=os.cpu_count(),
    )

    model = ChangeUnet(in_channels=3, out_channels=2, init_features=16)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    # 加入余弦退火算法
    # T_max = 10 代表每10个周期内变化学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
    loss_fn = nn.CrossEntropyLoss()


    # 余弦退火学习率优化，后期在使用
    # MAX_STEP = int(1e10)

    # tensorboard相关内容设置
    writer = SummaryWriter(log_dir='./log')

    epochs = 50
    for epoch in range(epochs):
       train_one_epoch(
           model=model,
           otpimizer=optimizer,
           scheduler=scheduler,
           loss_fn=loss_fn,
           train_loader=train_dl,
           val_loader=val_dl,
           device=device,
           epoch_idx=epoch
       )





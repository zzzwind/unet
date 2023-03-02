import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

transform = transforms.Compose(
    [
        transforms.ToTensor()
    ]
)

class ChangeNetDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.names = os.listdir(os.path.join(self.path, 'before', 'image'))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        before_img = Image.open(os.path.join(self.path, 'before', 'image', name))
        after_img = Image.open(os.path.join(self.path, 'after', 'image', name))
        label_img = Image.open(os.path.join(self.path, 'formatted_change_label', name.replace('jpg', 'png')))
        return transform(before_img), transform(after_img), torch.Tensor(np.array(label_img))

if __name__ == '__main__':
    dataset = ChangeNetDataset(path='/Users/jachin/development/deep_learning/unet/images')
    before, after , label = dataset[100]
    print('ss')





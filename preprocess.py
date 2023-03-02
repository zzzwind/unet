import glob
import os.path

import cv2 as cv
import numpy as np
from PIL import Image
from torchvision import transforms


# 将一张大图等距离分割成小图
def crop_img():
    img = cv.imread(
        '/Users/jachin/Downloads/Building change detection dataset/1. The two-period image data/after/after_label.tif')
    side = 256  # 裁剪大小256*256
    num_h = img.shape[0] // side
    num_w = img.shape[1] // side
    img = np.array(img)
    # img_gt = np.array(img_gt)
    img_crop = np.zeros((256, 256, 3))
    image = []

    for h in range(0, num_h):
        for w in range(0, num_w):
            img_crop = img[h * 256:(h + 1) * 256, w * 256:(w + 1) * 256]
            image.append(img_crop)
            pass
        pass

    path_img = 'images/after/label/'  # 保存路径
    for i in range(0, len(image)):
        image_i = image[i]

        path_image_i = path_img + str(i + 1) + str('.png')
        cv.imwrite(path_image_i, image_i)
# 将标签转换成二值图片
def convert(path):
   for file in  glob.glob(path):
       img = Image.open(file)



if __name__ == '__main__':
    # crop_img()
    # img = Image.open('images/change_label/85.png')
    # img = np.array(img)
    # x = transforms.Grayscale(num_output_channels=1)
    # print("sss")
    convert('images/change_label/*.png')

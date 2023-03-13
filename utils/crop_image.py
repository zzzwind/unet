import os.path

import cv2 as cv
import numpy as np
from pathlib import Path


def crop_image(image_path, save_path):
    img = cv.imread(str(image_path))
    image_name = image_path.stem
    size = img.shape
    w = size[1]  # 宽度
    h = size[0]  # 高度
    new_size = 256  # 裁剪大小256*256
    num_h = h // new_size
    num_w = w // new_size
    img_gt = np.array(img)
    img_crop = np.zeros((new_size, new_size, 3))
    images = []
    for h in range(0, num_h):
        for w in range(0, num_w):
            img_crop = img[h * new_size:(h + 1) * new_size, w * new_size:(w + 1) * new_size]
            images.append(img_crop)
            pass
        pass

    for i in range(0, len(images)):
        image_i = images[i]
        image_path_i = save_path / f'{image_name}-{i}.png'
        cv.imwrite(str(image_path_i), image_i)
    print("处理完毕")


if __name__ == "__main__":
    image_path = r"F:\dataset\LEVIR-CD\val\label"
    save_path = r"F:\dataset\LEVIR-CD\changed-path\label"
    count = 0
    for file in Path(image_path).glob('*.png'):
        crop_image(file, Path(save_path))
        count += 1
        print(f'处理了 {count} 张照片')

    # ss = Path(image_path)


    # crop_image(image_path=Path(image_path), save_path=Path(save_path))
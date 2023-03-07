from torchvision import transforms
from PIL import Image
from models.unet import ChangeUnet
import torch
import matplotlib.pyplot as plt

def show_pic(path, name):

    image = plt.imread(path)
    plt.imshow(image, cmap='gray')
    plt.title(name)
    plt.show()

if __name__ == '__main__':

    transform = transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )
    model = ChangeUnet()
    model.load_state_dict(torch.load('weight/change-net/change-net.pth', map_location='cpu'))

    before_image_pic = 'images/before/image/317.jpg'
    after_image_pic = 'images/after/image/317.jpg'


    change_label_name = 'images/change_label/317.png'
    before_label = 'images/before/label/317.png'
    after_label = 'images/after/label/317.png'

    before_image = Image.open(before_image_pic)
    after_image = Image.open(after_image_pic)

    before_image = transform(before_image)
    after_image = transform(after_image)

    before_image = before_image.unsqueeze(0)
    after_image = after_image.unsqueeze(0)

    out = model(before_image, after_image)
    out = torch.argmax(out, dim=1)
    out = out.permute((1, 2, 0))
    # out = out * 255
    out = out.numpy()
    plt.imshow(out, cmap='gray')
    plt.title('out-317')
    plt.show()

    show_pic(change_label_name, "change-317")
    show_pic(before_label, "before-317")
    show_pic(after_label, "after-317")



    print("over")

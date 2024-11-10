import cv2
from PIL import Image
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import transforms

from pretrain import prepreprocess_image, preprocess_image
from train import classes, HealthModel, device

loader = transforms.Compose([
    transforms.Resize((250, 250)),
    transforms.ToTensor()
])


def image_loader(image):
    """load image, returns cuda tensor"""
    image = Image.fromarray(image, mode="RGB")
    image = loader(image).float()
    image = image.unsqueeze(0)
    return image


def predict(image, _model):
    image = image_loader(image)

    with torch.no_grad():
        image.to(device)

        _model.eval()
        output = model(image)

        index = output.data.cpu().numpy().argmax()
        ps = output.data.cpu().numpy()

    return classes[index], ps[0].max()


if __name__ == "__main__":
    drive_img = cv2.imread(
        r"C:\Users\carso\Downloads\devnet-sample-images"
        r"\image_2024-09-29_08-58-55_orange_pepper.jpg")
    drive_img = cv2.cvtColor(drive_img, cv2.COLOR_BGR2RGB)
    drive_img = prepreprocess_image(drive_img, __debug=True)
    print("Image processed")

    num_img = len(drive_img)
    width = int(np.ceil(np.sqrt(num_img)))

    model = HealthModel()
    model.load_state_dict(torch.load("model.pth", weights_only=True))
    print("Model loaded")

    font = {
        "size": 9
    }

    for i in range(num_img):
        leaf = drive_img[i]
        plt.subplot(width, width, i + 1)
        leaf = preprocess_image(leaf)
        plt.imshow(leaf)
        plt.xticks([])
        plt.yticks([])
        pred = predict(leaf, model)
        plt.title(f"{pred[0]} {abs(pred[1] - 0.5) * 200:.2f}%", fontdict=font, y=0.95)

    plt.show()

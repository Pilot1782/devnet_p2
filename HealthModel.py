from typing import Union

import numpy as np
import requests
import torch
from PIL import Image
from numpy import ndarray
from torch import Tensor
from torchvision.transforms import transforms

from pretrain import preprocess_image, prepreprocess_image
from train import HealthNetwork


class HealthModel:
    def __init__(self, weights: str):
        """
        Initializes the Health Model
        Args:
            weights: path to weights file, ex: "~/model.pth"
        """
        self.__model = HealthNetwork()
        self.__loader = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.ToTensor()
        ])

        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        __checkpoint = torch.load(weights, weights_only=True)
        self.__model.load_state_dict(__checkpoint)
        self.__model = self.__model.to(self.__device)

    def _image_loader(self, image: ndarray) -> Tensor:
        """
        Loads an image
        Args:
            image: the image as an rgb ndarray

        Returns:
            The tensor
        """
        image = Image.fromarray(image, mode="RGB")
        image = self.__loader(image).float()
        image = image.unsqueeze(0)
        return image.cuda()

    def _predict(self, image):
        self.__model.eval()
        with torch.no_grad():
            image.to(self.__device)

            output = self.__model(image)
            probs = torch.nn.functional.softmax(output, dim=1)

            ps, index = torch.max(probs, 1)
            ps = ps.item()
            index = index.item()

        return float(index), float(ps)

    def predict(self, image: Union[str, ndarray], multi_leaf=True, __debug=False) -> tuple[str, float]:
        """
        Predicts the health of the plant

        Args:
            image: path to image or a numpy array of the image (must be RGB)
            multi_leaf: whether to use multi leaf prediction or not
            __debug: whether to show debugging images

        Returns:
            class, confidence: The health of the plant as either "healthy" or "unhealthy" and the confidence in the prediction
        """

        if type(image) is str and image.split(".")[-1] in ("png", "jpg", "webp"):
            image = np.array(requests.get(image, stream=True).raw)

        images = [image]
        if multi_leaf:
            images = prepreprocess_image(image, __debug=__debug)

        images = [self._image_loader(preprocess_image(image)) for image in images]

        healths = np.arange(0, len(images))
        confs = np.array(list(range(len(images))), dtype=np.float64)

        for i in range(len(images)):
            health, confidence = self._predict(images[i])
            healths[i] = health
            confs[i] = confidence

        avg_health = round(np.average(healths))
        avg_confidence = float(np.average(confs))

        return ("healthy", "unhealthy")[avg_health], avg_confidence


if __name__ == '__main__':
    model = HealthModel('weights.pth')
    print(model.predict(r"C:\Users\carso\Downloads\devnet-sample-images\image_2024-10-06_12-30-03_orange_pepper.jpg"))

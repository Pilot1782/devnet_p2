from typing import Union

import numpy as np
import requests
import torch
from numpy import ndarray
from torch import Tensor
from torchvision.transforms import transforms

from pretrain import preprocess_image, prepreprocess_image
from train import HealthNetwork, classes


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

    def _image_loader(self, image: ndarray) -> Tensor:
        """
        Loads an image
        Args:
            image: the image as an rgb ndarray

        Returns:
            The tensor
        """
        image = self.__loader(image).float()
        image = image.unsqueeze(0)
        return image.cuda()

    def predict(self, image: Union[str, ndarray], multi_leaf=True) -> tuple[str, float]:
        """
        Predicts the health of the plant

        Args:
            image: path to image or a numpy array of the image (must be RGB)
            multi_leaf: whether to use multi leaf prediction or not

        Returns:
            class, confidence: The health of the plant as either "healthy" or "unhealthy" and the confidence in the prediction
        """

        if type(image) is str:
            image = np.array(requests.get(image, stream=True).raw)

        images = [image]
        if multi_leaf:
            images = prepreprocess_image(image)

        images = [self._image_loader(preprocess_image(image)) for image in images]

        predictions = []

        for image in images:
            with torch.no_grad():
                self.__model.eval()
                out = self.__model(image)
                pred = out.data.cpu().numpy()

                health = pred.argmax()
                confidence = pred[pred.max()]

                predictions.append((health, confidence))

        avg_health = round(sum((i[0] for i in predictions)) / len(predictions))
        avg_confidence = round(sum((i[1] for i in predictions)) / len(predictions))

        return classes[avg_health], avg_confidence


if __name__ == '__main__':
    model = HealthModel('weights.pth')
    print(model.predict())

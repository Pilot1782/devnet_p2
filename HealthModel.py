from typing import Union, Tuple

import cv2
import numpy as np
import requests
import torch
from PIL import Image
from matplotlib import pyplot as plt
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
        __checkpoint = torch.load(weights, weights_only=True, map_location=self.__device)
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
        if self.__device == 'cuda':
            return image.cuda()
        return image.cpu()

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

    def _predict_alg(self, image):
        """
        Algorithmic prediction of a model
        Args:
            image:

        Returns: bool (True: healthy, False: not healthy)

        """
        g, y, b = cv2.split(image)

        g = np.sum(g)
        y = np.sum(y)
        b = np.sum(b)

        return 5 >= (y + b)

    def predict(self, image: Union[str, ndarray], multi_leaf=True, _debug=False) -> tuple[int, float]:
        """
        Predicts the health of the plant

        Args:
            image: path to image or a numpy array of the image (must be RGB)
            multi_leaf: whether to use multi leaf prediction or not
            _debug: whether to show debugging images

        Returns:
            class, confidence: The health of the plant as either "healthy" or "unhealthy" and the confidence in the prediction
        """

        if type(image) is str and image.split(".")[-1] in ("png", "jpg", "webp"):
            image = np.array(requests.get(image, stream=True).raw)

        images = [image]
        if multi_leaf:
            images = prepreprocess_image(image, __debug=_debug)

        images = [preprocess_image(image) for image in images]
        alg_preds = [self._predict_alg(image) for image in images]

        while len(images) > 2:
            cur = images[-1]
            if np.sum(images[-2] - cur) < 1:
                images = images[:-1]
            else:
                break

        if _debug:
            sq = int(np.ceil(np.sqrt(len(images))))
            for i in range(len(images)):
                plt.subplot(sq, sq, i + 1)
                plt.imshow(images[i])
                if alg_preds[i]:
                    plt.axis('off')
            plt.show()

            cv2.imwrite("currentSavedImage.jpg", images[-1])

        images = [self._image_loader(img) for img in images]

        healths = np.arange(-1, len(images))
        confs = np.array(list(range(len(images))), dtype=np.float64)

        for i in range(len(images)):
            health, confidence = self._predict(images[i])
            if len(images) > 2 and alg_preds[i] and health >= 0.5:
                health = 0.0

            healths[i] = health
            confs[i] = confidence

        healths = healths[:-1]

        avg_health = round(np.average(healths))
        avg_confidence = float(np.average(confs))

        return avg_health, avg_confidence

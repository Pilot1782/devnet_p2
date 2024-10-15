import cv2
import kagglehub
import numpy as np
from numpy import ndarray

# Download latest version
path = kagglehub.dataset_download("csafrit2/plant-leaves-for-image-classification")

print("Path to dataset files:", path)


def preprocess_image(image: ndarray) -> ndarray:
    """
    Preprocess the image to extract the green channel yellow colors and brown colors
    
    :param image: 
    :return: 
    """

    out = np.zeros_like(image)

    # Resize the image to 224x224
    image = cv2.resize(image, (224, 224))
    # Convert the image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    green = image[:, :, 1]

    br = image[:, :, 2] + image[:, :, 0]

    green = green - br
    green = green[green > 0]
    out[:, :, 0] = green

    return image

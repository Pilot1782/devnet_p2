import cv2
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

# Download latest version
path = kagglehub.dataset_download("csafrit2/plant-leaves-for-image-classification")

print("Path to dataset files:", path)


def preprocess_image(image: ndarray) -> tuple[ndarray]:
    """
    Preprocess the image to extract the green channel yellow colors and brown colors

    :param image: Image as a 3 channel ndarray
    :return: Image as a 3 channel (green, yellow, brown) ndarray
    """

    # Resize the image to 224x224
    image = cv2.resize(image, (250, 250))
    out = np.zeros_like(image)
    # Convert the image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    green = image[:, :, 1]

    br = image[:, :, 2] + image[:, :, 0]

    green = green - br
    green = np.clip(green, 100, 255)
    out[:, :, 0] = green

    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # hsv range: H: +/- 10, S: +/- 10, V: +/- 10

    yellow = cv2.inRange(image, np.array([33, 84, 62]), np.array([53, 100, 82]))

    out[:, :, 1] = yellow

    brown = cv2.inRange(image, np.array([15, 0, 90]), np.array([35, 17, 100]))

    out[:, :, 2] = brown

    return out, green, yellow, brown


img = cv2.imread(
    path +
    r"\Plants_2\test"
    r"\Basil healthy (P8)\0008_0001.JPG"
)

out = preprocess_image(img)

plt.imshow(out[1], cmap="gray")
plt.show()

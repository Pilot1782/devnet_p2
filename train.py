import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

# Download latest version
# path = kagglehub.dataset_download("csafrit2/plant-leaves-for-image-classification")
path = r"C:\Users\pilot1784\.cache\kagglehub\datasets\csafrit2\plant-leaves-for-image-classification\versions\2"

print("Path to dataset files:", path)


def laplace_of_gaussian(gray_img, sigma=1., kappa=0.75, pad=False):
    """
    Applies Laplacian of Gaussians to grayscale image.

    :param gray_img: image to apply LoG to
    :param sigma:    Gauss sigma of Gaussian applied to image, <= 0. for none
    :param kappa:    difference threshold as factor to mean of image values, <= 0 for none
    :param pad:      flag to pad output w/ zero border, keeping input image size
    """
    assert len(gray_img.shape) == 2
    img = cv2.GaussianBlur(gray_img, (0, 0), sigma) if 0. < sigma else gray_img
    img = cv2.Laplacian(img, cv2.CV_64F)
    rows, cols = img.shape[:2]
    # min/max of 3x3-neighbourhoods
    min_map = np.minimum.reduce(list(img[r:rows - 2 + r, c:cols - 2 + c]
                                     for r in range(3) for c in range(3)))
    max_map = np.maximum.reduce(list(img[r:rows - 2 + r, c:cols - 2 + c]
                                     for r in range(3) for c in range(3)))
    # bool matrix for image value positiv (w/out border pixels)
    pos_img = 0 < img[1:rows - 1, 1:cols - 1]
    # bool matrix for min < 0 and 0 < image pixel
    neg_min = min_map < 0
    neg_min[1 - pos_img] = 0
    # bool matrix for 0 < max and image pixel < 0
    pos_max = 0 < max_map
    pos_max[pos_img] = 0
    # sign change at pixel?
    zero_cross = neg_min + pos_max
    # values: max - min, scaled to 0--255; set to 0 for no sign change
    value_scale = 255. / max(1., img.max() - img.min())
    values = value_scale * (max_map - min_map)
    values[1 - zero_cross] = 0.
    # optional thresholding
    if 0. <= kappa:
        thresh = float(np.absolute(img).mean()) * kappa
        values[values < thresh] = 0.
    log_img = values.astype(np.uint8)
    if pad:
        log_img = np.pad(log_img, pad_width=1, mode='constant', constant_values=0)
    return log_img


def contrast_stretching(_img):
    """
    Apply contrast stretching to the image

    :param _img: Image as a 3 channel ndarray
    :return: Image as a 3 channel ndarray
    """
    # Apply CLAHE to the V channel
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
    _img = clahe.apply(_img)
    return _img


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

    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    plt.subplot(2, 3, 1, title="Original image")
    plt.imshow(image)

    green = np.clip(g - r - b, 0, 255)
    green = cv2.fastNlMeansDenoising(green, None, 4, 7, 21)

    edges = laplace_of_gaussian(green, cv2.CV_8U)
    edges = contrast_stretching(edges)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.blur(edges, (3, 3))
    edges = laplace_of_gaussian(edges, cv2.CV_8U)

    plt.subplot(2, 3, 4, title="Edges")
    plt.imshow(edges, cmap="gray")
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(green)

    cv2.fillPoly(mask, [max_area], 255)

    plt.subplot(2, 3, 5, title="Green Mask")
    plt.imshow(mask, cmap="gray")

    green = green * mask

    # hsv range: H: +/- 10, S: +/- 10, V: +/- 10
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    out[:, :, 0] = green

    yellow = cv2.inRange(image, np.array([18, 0, 0]), np.array([38, 255, 255]))  # 56->28
    yellow = cv2.bitwise_and(image, image, mask=yellow)
    yellow = cv2.cvtColor(yellow, cv2.COLOR_HSV2RGB)
    yellow = np.average(yellow, axis=2)

    out[:, :, 1] = yellow

    brown = cv2.inRange(image, np.array([5, 0, 0]), np.array([25, 255, 255]))  # 30->15
    brown = cv2.bitwise_and(image, image, mask=brown)
    brown = cv2.cvtColor(brown, cv2.COLOR_HSV2RGB)
    brown = np.average(brown, axis=2)

    out[:, :, 2] = brown

    plt.subplot(2, 3, 2, title="Filtered Green")
    plt.imshow(green, cmap="gray")
    plt.subplot(2, 3, 3, title="Filtered Yellow")
    plt.imshow(yellow, cmap="gray")
    plt.subplot(2, 3, 6, title="Filtered Brown")
    plt.imshow(brown, cmap="gray")

    return out, green, yellow, brown


img = cv2.imread(
    path +
    r"\Plants_2\test"
    r"\Bael diseased (P4b)\0016_0010.JPG"
)

out = preprocess_image(img)

plt.show()

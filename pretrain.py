# Download latest version
import os
import shutil

import cv2
import kagglehub
import numpy as np

path = kagglehub.dataset_download("csafrit2/plant-leaves-for-image-classification")

path += r"\Plants_2"

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


def preprocess_image(_image: np.ndarray) -> np.ndarray:
    """
    Preprocess the image to extract the green channel yellow colors and brown colors

    :param _image: Image as a 3 channel ndarray
    :return: Image as a 3 channel (green, yellow, brown) ndarray
    """

    # Resize the image to 224x224
    _image = cv2.resize(_image, (250, 250))
    _image = np.uint8(_image)
    out = np.zeros_like(_image)

    # hsv range: H: +/- 10, S: +/- 10, V: +/- 10
    _image = cv2.cvtColor(_image, cv2.COLOR_RGB2HSV)

    green = cv2.inRange(_image, np.array([35, 50, 50]), np.array([75, 255, 255]))  # 115-> 55
    green = cv2.bitwise_and(_image, _image, mask=green)
    green = cv2.cvtColor(green, cv2.COLOR_HSV2RGB)
    green = np.average(green, axis=2)

    out[:, :, 0] = green

    yellow = cv2.inRange(_image, np.array([18, 50, 50]), np.array([38, 255, 255]))  # 56->28
    yellow = cv2.bitwise_and(_image, _image, mask=yellow)
    yellow = cv2.cvtColor(yellow, cv2.COLOR_HSV2RGB)
    yellow = np.average(yellow, axis=2)

    out[:, :, 1] = yellow

    brown = cv2.inRange(_image, np.array([5, 50, 50]), np.array([25, 255, 255]))  # 30->15
    brown = cv2.bitwise_and(_image, _image, mask=brown)
    brown = cv2.cvtColor(brown, cv2.COLOR_HSV2RGB)
    brown = np.average(brown, axis=2)

    out[:, :, 2] = brown

    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    out = np.float64(out)
    out = out / 255.0

    return out


if __name__ == "__main__":
    for dirs in os.listdir(path):
        if not os.path.isdir(os.path.join(path, dirs, "healthy")):
            os.mkdir(os.path.join(path, dirs, "healthy"))

        if not os.path.isdir(os.path.join(path, dirs, "unhealthy")):
            os.mkdir(os.path.join(path, dirs, "unhealthy"))

        for plants in os.listdir(os.path.join(path, dirs)):
            if "diseased" in plants.lower():
                print("Copying", os.path.join(dirs, plants), "to unhealthy")

                for image in os.listdir(os.path.join(path, dirs, plants)):
                    if os.path.exists(os.path.join(path, dirs, "unhealthy", image)):
                        os.remove(os.path.join(path, dirs, "unhealthy", image))

                    shutil.copyfile(
                        os.path.join(path, dirs, plants, image),
                        os.path.join(path, dirs, "unhealthy", image)
                    )

                    img = cv2.imread(os.path.join(path, dirs, "unhealthy", image))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = preprocess_image(img)
                    img = np.uint8(img * 255.0)
                    cv2.imwrite(os.path.join(path, dirs, "unhealthy", image), img)
            elif (
                    "healthy" in plants.lower()
                    and "unhealthy" not in plants.lower()
                    and plants.lower() != "healthy"
            ):
                print("Copying", os.path.join(dirs, plants), "to healthy")

                for image in os.listdir(os.path.join(path, dirs, plants)):
                    if os.path.exists(os.path.join(path, dirs, "healthy", image)):
                        os.remove(os.path.join(path, dirs, "healthy", image))

                    shutil.copyfile(
                        os.path.join(path, dirs, plants, image),
                        os.path.join(path, dirs, "healthy", image)
                    )

                    img = cv2.imread(os.path.join(path, dirs, "healthy", image))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = preprocess_image(img)
                    img = np.uint8(img * 255.0)
                    cv2.imwrite(os.path.join(path, dirs, "healthy", image), img)
            else:
                print("Skipping", os.path.join(dirs, plants))

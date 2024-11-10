# Download latest version
import os
import shutil

import cv2
import kagglehub
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    path = kagglehub.dataset_download("csafrit2/plant-leaves-for-image-classification")
else:
    path = "..."

path += r"\Plants_2"


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

    out = np.float64(out)
    out = out / 255.0

    return out


def white_balance(_img):
    result = cv2.cvtColor(_img, cv2.COLOR_RGB2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
    return result


def water_split(_image: np.ndarray) -> list[np.ndarray]:
    """
    Splits the contour into separate contours. By watershed splitting
    Args:
        _image: the image to split as a monochrome image

    Returns: list of contours
    """

    out = []

    _image = np.uint8(_image)
    filled = np.zeros_like(_image)
    contour, _ = cv2.findContours(_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contour[0]
    cv2.drawContours(filled, [contour], -1, 255, -1)

    section_area = cv2.contourArea(contour)

    for i in range(1000):
        k = np.ones((i, i), np.uint8)
        eroded = cv2.erode(filled, k)

        if np.sum(eroded) == 0:
            print("Erosion complete")
            break

        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 1:
            total_area = 0
            before_area = 0

            for cnt in contours:
                before_area += cv2.contourArea(cnt)

                full = np.zeros_like(filled)
                cv2.drawContours(full, [cnt], -1, 255, -1)

                full = cv2.dilate(full, k)
                cnt, _ = cv2.findContours(full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                cnt_area = cv2.contourArea(cnt[0])
                total_area += cnt_area

                if cnt_area * 1.25 < section_area / len(contours):
                    continue

                out.append(cnt[0])

            if len(out) > 1:
                break
            out = []

    return out


def crop_to_content(_image: np.ndarray) -> np.ndarray:
    """
    Removes and black bars around the edges
    
    :param _image: some image as a ndarray
    :return: the image after cropping
    """

    gray = cv2.cvtColor(_image, cv2.COLOR_RGB2GRAY)

    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])

    return _image[y:y + h, x:x + w]


def is_closed(contour):
    return cv2.contourArea(contour) > cv2.arcLength(contour, True)


def prepreprocess_image(_image: np.ndarray, __debug=True) -> tuple[np.ndarray]:
    """
    Takes an overhead image of a plant box and crops to the leaves in the image

    :param __debug: Whether debugging images should be shown
    :param _image: 3 channel RGB image
    :return: list of 3 channel RGB images cropped to the leaves
    """

    # Step One: White Balance
    _image = white_balance(_image)

    if __debug:
        plt.subplot(2, 3, 1)
        plt.imshow(_image)
        plt.xticks([])
        plt.yticks([])
        plt.title("Orig")

    # add a 5px black border
    r = np.pad(_image[:, :, 0], 5, mode='constant', constant_values=0)
    g = np.pad(_image[:, :, 1], 5, mode='constant', constant_values=0)
    b = np.pad(_image[:, :, 2], 5, mode='constant', constant_values=0)
    _image.resize((r.shape[0], r.shape[1], 3))
    _image[:, :, 0] = r
    _image[:, :, 1] = g
    _image[:, :, 2] = b

    _image = cv2.cvtColor(_image, cv2.COLOR_RGB2HSV)

    # Step Two: Find Green Areas
    green = cv2.inRange(_image, np.array([35, 50, 40]), np.array([75, 255, 255]))
    green = cv2.bitwise_and(_image, _image, mask=green)
    green = cv2.cvtColor(green, cv2.COLOR_HSV2RGB)

    if __debug:
        plt.subplot(2, 3, 2)
        plt.imshow(green)
        plt.xticks([])
        plt.yticks([])
        plt.title("Green")

    # Step Three: Divide The Image Into Leaves
    leaves = cv2.threshold(green, 0, 255, cv2.THRESH_BINARY)[1]
    leaf_edges = cv2.Canny(leaves, 100, 200)
    leaf_edges = cv2.dilate(leaf_edges, np.ones((3, 3), np.uint8), iterations=1)

    if __debug:
        plt.subplot(2, 3, 3)
        plt.imshow(leaf_edges)
        plt.xticks([])
        plt.yticks([])
        plt.title("Edges")

    contours, _ = cv2.findContours(leaf_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tmp = ()

    for contour in contours:
        if not is_closed(contour):
            contour = cv2.approxPolyDP(contour, 0.001 * cv2.arcLength(contour, True), True)
        tmp += (contour,)

    tmp = np.zeros_like(_image)
    cv2.drawContours(tmp, contours, -1, (255, 255, 255), -1)

    if __debug:
        plt.subplot(2, 3, 4)
        plt.imshow(tmp)
        plt.xticks([])
        plt.yticks([])
        plt.title("Contours")

    has_multi_leaf = True
    fixed_cnts = contours
    while has_multi_leaf:
        has_multi_leaf = False

        tmp_contours = ()
        for contour in fixed_cnts:
            if not is_closed(contour):
                continue

            area = cv2.contourArea(contour)
            if area < 1000:  # Filter out small contours
                continue

            thresh = 0.5

            hull = cv2.convexHull(contour)
            pdiff = (cv2.contourArea(hull) - area) / area

            if pdiff > thresh:
                if __debug:
                    print(f"Percent diff of areas: {pdiff * 100:.2f}% {'FLAGGED' if pdiff > thresh else ''}")

                has_multi_leaf = True
                tmp = np.zeros_like(green)
                cv2.drawContours(tmp, [contour], -1, (255, 255, 255), -1)
                tmp = np.average(tmp, 2)
                split = water_split(tmp)
                for s in split:
                    tmp_contours += (s,)
            else:
                tmp_contours += (contour,)

        fixed_cnts = tmp_contours

    out = []
    tmp = np.zeros_like(_image)
    for contour in fixed_cnts:
        mask = np.zeros_like(green)
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
        cv2.drawContours(tmp, [contour], -1, (np.random.randint(255), np.random.randint(255), np.random.randint(255)),
                         -1)
        mask = np.uint8(np.average(mask, 2))

        _leaf = cv2.bitwise_and(_image, _image, mask=mask)
        _leaf = cv2.cvtColor(_leaf, cv2.COLOR_HSV2RGB)
        _leaf = crop_to_content(_leaf)
        out.append(_leaf)

    if __debug:
        plt.subplot(2, 3, 5)
        plt.imshow(tmp)
        plt.xticks([])
        plt.yticks([])
        plt.title("Final")
        plt.show()

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

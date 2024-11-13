import os
import random
from argparse import ArgumentParser

import cv2

from HealthModel import HealthModel

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-p", "--path",
        help="The path to the image or image directory"
    )
    parser.add_argument(
        "-w", "--weights",
        help="The path to the weights of the model",
        default=os.path.join(os.getcwd(), 'model.pth')
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to enable very verbose and helpful debugging"
    )
    args = parser.parse_args()

    model = HealthModel(args.weights)

    if os.path.isdir(args.path):
        imgs = os.listdir(args.path)
    else:
        path = args.path.split(os.sep)
        args.path = os.sep.join(path[:-1])
        imgs = [path[-1]]

    random.shuffle(imgs)
    print(imgs)
    for img in imgs:
        if img.split(".")[-1].lower() not in ("bmp dib jpeg jpg jpe jp2 png webp "
                                      "avif pbm pgm ppm pxm pnm pfm sr "
                                      "ras tiff tif exr hdr pic").split(" "):
            continue

        image = cv2.imread(os.path.join(args.path, img))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pred = model.predict(image, multi_leaf=not img[0].isdigit(), _debug=args.debug)
        print(f"{img}: {pred[0]} ({pred[1] * 100:.2f}%)")

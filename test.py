import os
import random

import cv2

from HealthModel import HealthModel

if __name__ == "__main__":
    model = HealthModel(os.path.join(os.getcwd(), 'model.pth'))

    imgs = os.listdir(r"C:\Users\carso\Downloads\devnet-sample-images")
    random.shuffle(imgs)
    for img in imgs[:5]:
        image = cv2.imread("C:\\Users\\carso\\Downloads\\devnet-sample-images\\" + img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pred = model.predict(image, multi_leaf=not img[0].isdigit(), _debug=True)
        print(f"{img}: {pred[0].upper()} ({pred[1] * 100:.2f}%)")

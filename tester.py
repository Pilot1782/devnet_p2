import os

import kagglehub
import numpy as np
from keras.api.models import load_model
from keras.api.preprocessing.image import load_img

from pretrain import preprocess_image

model = load_model(os.path.join(os.path.dirname(os.path.realpath(__file__)), "model_saved.keras"))

path = kagglehub.dataset_download("csafrit2/plant-leaves-for-image-classification")

wrong = 0
right = 0
print('\n' + '=' * 8 + ' UNHEALTHY ' + '=' * 8)

for image in os.listdir(os.path.join(path, "Plants_2", "test", "unhealthy")):
    img = load_img(os.path.join(path, "Plants_2", "test", "unhealthy", image), target_size=(250, 250))
    img = np.array(img)
    img = img.reshape(1, 250, 250, 3)
    result = model.predict(img, verbose=0)[0][0]
    if result < 0.5:
        wrong += 1
    else:
        right += 1

    print(
        f"Image: {image}, "
        f"Result: {'  healthy' if result < 0.5 else 'unhealthy'} "
        f"({abs(0.5 - result) * 200:.2f}%)"
        f"{'  (WRONG)' if result < 0.5 else ''}"
    )

print('\n' + '=' * 8 + ' HEALTHY ' + '=' * 8)

for image in os.listdir(os.path.join(path, "Plants_2", "test", "healthy")):
    img = load_img(os.path.join(path, "Plants_2", "test", "healthy", image), target_size=(250, 250))
    img = np.array(img)
    img = img.reshape(1, 250, 250, 3)
    result = model.predict(img, verbose=0)[0][0]
    if result > 0.5:
        wrong += 1
    else:
        right += 1

    print(
        f"Image: {image}, "
        f"Result: {'  healthy' if result < 0.5 else 'unhealthy'} "
        f"({abs(0.5 - result) * 200:.2f}%)"
        f"{'  (WRONG)' if result > 0.5 else ''}"
    )

print('\n' + '=' * 8 + ' UNKNOWN ' + '=' * 8)

if os.path.exists(r"C:\Users\pilot1784\Downloads\drive-download-20241016T233706Z-001"):
    for image in os.listdir(r"C:\Users\pilot1784\Downloads\drive-download-20241016T233706Z-001"):
        img = load_img(os.path.join(r"C:\Users\pilot1784\Downloads\drive-download-20241016T233706Z-001", image),
                       target_size=(250, 250))
        img = np.array(img)
        img = preprocess_image(img)
        img = img.reshape(1, 250, 250, 3)
        result = model.predict(img, verbose=0)[0][0]

        print(
            f"Image: {image}, "
            f"Result: {'  healthy' if result < 0.5 else 'unhealthy'} "
            f"({abs(0.5 - result) * 200:.2f}%)"
        )

print(f"\nWrong predictions: {wrong}")
print(f"Accuracy: {(right + wrong) - wrong / 2:.2f}%")

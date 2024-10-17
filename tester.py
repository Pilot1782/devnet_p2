import os

import cv2
import torch
import kagglehub
import numpy as np

from pretrain import preprocess_image

# model = load_model(os.path.join(os.path.dirname(os.path.realpath(__file__)), "model_saved.keras"))

path = kagglehub.dataset_download("csafrit2/plant-leaves-for-image-classification")


print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Storing ID of current CUDA device
cuda_id = torch.cuda.current_device()
print(f"ID of current CUDA device: {torch.cuda.current_device()}")

print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")

# 
# wrong = 0
# right = 0
# print('\n' + '=' * 8 + ' UNHEALTHY ' + '=' * 8)
# 
# for image in os.listdir(os.path.join(path, "Plants_2", "test", "unhealthy")):
#     img = load_img(os.path.join(path, "Plants_2", "test", "unhealthy", image), target_size=(250, 250))
#     img = np.array(img)
#     img = img.reshape(1, 250, 250, 3)
#     result = model.predict(img, verbose=0)[0][0]
#     if result < 0.5:
#         wrong += 1
#     else:
#         right += 1
# 
#     print(
#         f"Image: {image}, "
#         f"Result: {'  healthy' if result < 0.5 else 'unhealthy'} "
#         f"({abs(0.5 - result) * 200:.2f}%)"
#         f"{'  (WRONG)' if result < 0.5 else ''}"
#     )
# 
# print('\n' + '=' * 8 + ' HEALTHY ' + '=' * 8)
# 
# for image in os.listdir(os.path.join(path, "Plants_2", "test", "healthy")):
#     img = load_img(os.path.join(path, "Plants_2", "test", "healthy", image), target_size=(250, 250))
#     img = np.array(img)
#     img = img.reshape(1, 250, 250, 3)
#     result = model.predict(img, verbose=0)[0][0]
#     if result > 0.5:
#         wrong += 1
#     else:
#         right += 1
# 
#     print(
#         f"Image: {image}, "
#         f"Result: {'  healthy' if result < 0.5 else 'unhealthy'} "
#         f"({abs(0.5 - result) * 200:.2f}%)"
#         f"{'  (WRONG)' if result > 0.5 else ''}"
#     )
# 
# print('\n' + '=' * 8 + ' UNKNOWN LEAVES ' + '=' * 8)
# 
# for image in os.listdir(os.path.join(path, "Plants_2", "images to predict")):
#     if not image.lower().endswith(".jpg"):
#         continue
# 
#     img = cv2.imread(os.path.join(path, "Plants_2", "images to predict", image))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = preprocess_image(img)
#     img = img.reshape(1, 250, 250, 3)
#     result = model.predict(img, verbose=0)[0][0]
# 
#     print(
#         f"Image: {image}, "
#         f"Result: {'  healthy' if result < 0.5 else 'unhealthy'} "
#         f"({abs(0.5 - result) * 200:.2f}%)"
#     )
# 
# print('\n' + '=' * 8 + ' UNKNOWN PLANT BOX ' + '=' * 8)
# 
# if os.path.exists(r"C:\Users\pilot1784\Downloads\drive-download-20241016T233706Z-001"):
#     for image in os.listdir(r"C:\Users\pilot1784\Downloads\drive-download-20241016T233706Z-001"):
#         img = cv2.imread(os.path.join(r"C:\Users\pilot1784\Downloads\drive-download-20241016T233706Z-001", image))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = preprocess_image(img)
#         img = img.reshape(1, 250, 250, 3)
#         result = model.predict(img, verbose=0)[0][0]
# 
#         print(
#             f"Image: {image}, "
#             f"Result: {'  healthy' if result < 0.5 else 'unhealthy'} "
#             f"({abs(0.5 - result) * 200:.2f}%)"
#         )
# 
# print(f"\nWrong predictions: {wrong}")
# print(f"Accuracy: {(right + wrong) - wrong / 2:.2f}%")

import cv2

from HealthModel import HealthModel

if __name__ == "__main__":
    model = HealthModel('weights.pth')

    imgs = [
        r"C:\Users\carso\Downloads\devnet-sample-images\image_2024-10-06_12-30-02_orange_pepper.jpg",
        r"C:\Users\carso\Downloads\devnet-sample-images\image_2024-10-06_16-30-03_orange_pepper.jpg",
        r"C:\Users\carso\Downloads\devnet-sample-images\image_2024-10-06_08-30-02_orange_pepper.jpg"
    ]
    for img in imgs:
        image = cv2.imread(img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pred = model.predict(image)
        print(f"{pred[0].upper()} ({pred[1] * 100:.2f}%)")

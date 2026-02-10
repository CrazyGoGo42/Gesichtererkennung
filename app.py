import numpy as np
import cv2
import os

face_databank = np.zeros((30, 256, 256), dtype=np.uint8)

script_dir = os.path.dirname(os.path.abspath(__file__))
bilder_dir = os.path.join(script_dir, "Bilder")

for i in range(30):
    img_path = os.path.join(bilder_dir, f"{i+1}.jpg")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Konnte die Bilder nicht laden: {img_path}")
        continue

    img = cv2.resize(img, (256, 256))
    face_databank[i] = img


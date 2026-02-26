import numpy as np
import cv2
import os

face_databank = np.zeros((30, 512, 512), dtype=np.uint8)

script_dir = os.path.dirname(os.path.abspath(__file__))
bilder_dir = os.path.join(script_dir, "Bilder")

for i in range(30):
    img_path = os.path.join(bilder_dir, f"{i+1}.jpg")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Konnte die Bilder nicht laden: {img_path}")
        continue

    img = cv2.resize(img, (512, 512))
    face_databank[i] = img

np.save("data.npy", face_databank)

face_databank = np.load("data.npy")


binary_face_databank = np.zeros((30, 512, 512), dtype=np.uint8)

for i in range(30):
    _, binary_img = cv2.threshold(face_databank[i], 127, 255, cv2.THRESH_BINARY)
    binary_face_databank[i] = binary_img

np.save("binary_data.npy", binary_face_databank)

# for i in range(30):                                             Zeigt die Bilder
#     cv2.imshow(f"Binary Face {i+1}", binary_face_databank[i])
#     cv2.waitKey(0)
# cv2.destroyAllWindows()


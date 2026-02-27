import numpy as np
import cv2
import os

face_databank = np.zeros((30, 256, 256), dtype=np.uint8)

script_dir = os.path.dirname(os.path.abspath(__file__))
bilder_dir = os.path.join(script_dir, "Bilder_256")

for i in range(30):
    img_path = os.path.join(bilder_dir, f"{i+1}.jpg")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Konnte die Bilder nicht laden: {img_path}")
        continue

    img = cv2.resize(img, (256, 256))
    face_databank[i] = img

np.save("data.npy", face_databank)

face_databank = np.load("data.npy")


binary_face_databank = np.zeros((30, 256, 256), dtype=np.uint8)

for i in range(30):
    _, binary_img = cv2.threshold(face_databank[i], 127, 255, cv2.THRESH_BINARY)
    binary_face_databank[i] = binary_img

np.save("binary_data.npy", binary_face_databank)


def lbp(image):
    lbp_image = np.zeros_like(image, dtype=np.uint8)

    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            center = image[i, j]

            code = 0
            code |= (image[i-1, j-1] >= center) << 7
            code |= (image[i-1, j]   >= center) << 6
            code |= (image[i-1, j+1] >= center) << 5
            code |= (image[i, j+1]   >= center) << 4
            code |= (image[i+1, j+1] >= center) << 3
            code |= (image[i+1, j]   >= center) << 2
            code |= (image[i+1, j-1] >= center) << 1
            code |= (image[i, j-1]   >= center) << 0

            lbp_image[i, j] = code

    return lbp_image

# TEST!!!! imshow with range 3:
for i in range(3):
    lbp_image = lbp(face_databank[i])
    cv2.imshow(f"LBP Image {i+1}", lbp_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
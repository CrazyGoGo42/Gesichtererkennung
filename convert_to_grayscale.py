import cv2
import os
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
bilder_dir = os.path.join(script_dir, "Bilder")

# Supported image extensions
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

# Get all image files in the Bilder directory
image_files = [f for f in os.listdir(bilder_dir)
               if Path(f).suffix.lower() in image_extensions]

print(f"Found {len(image_files)} images to convert...")

for filename in image_files:
    img_path = os.path.join(bilder_dir, filename)

    # Read image in color
    img = cv2.imread(img_path)

    if img is None:
        print(f"Could not read: {filename}")
        continue

    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Save back to the same path (replaces original)
    cv2.imwrite(img_path, gray_img)
    print(f"Converted: {filename}")

print("Conversion complete!")


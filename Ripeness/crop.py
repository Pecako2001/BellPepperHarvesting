from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# Load the model
m = YOLO('best.pt')

# Assuming you have an image file, specify its path
image_path = 'Cropping.jpg'

# Use the model to predict on the specified image
res = m.predict(image_path)

# Iterate through detection results
for r in res:
    img = np.copy(r.orig_img)
    img_name = Path(r.path).stem

    # Iterate through each detected object and its contour
    for ci, c in enumerate(r):
        # Get the label of the detected object
        label = c.names[c.boxes.cls.tolist().pop()]

        # Initialize a binary mask for the detected object
        b_mask = np.zeros(img.shape[:2], np.uint8)

        # Create a contour mask
        contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
        _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

        # OPTION-1: Isolate object with black background
        mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
        isolated_black_bg = cv2.bitwise_and(mask3ch, img)

        # OPTION-2: Isolate object with transparent background (when saved as PNG)
        transparent_bg = np.dstack([img, b_mask])

        # Detection crop (can use either isolated_black_bg or transparent_bg)
        x1, y1, x2, y2 = c.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
        iso_crop = transparent_bg[y1:y2, x1:x2]  # Or replace with transparent_bg for the alternative

        # TODO: Here you can save the cropped image, display it, or perform other actions
        # For example, saving the isolated object with a black background
        crop_file_name = f"{img_name}_{label}_{ci}.png"
        cv2.imwrite(crop_file_name, iso_crop)

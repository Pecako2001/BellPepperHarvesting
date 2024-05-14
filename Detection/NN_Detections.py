from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

"""
This script demonstrates how to perform object segmentation and classification on a video using YOLOv8.
The script loads a video file, segments objects in each frame, and classifies the segmented objects.
The segmentation model is used to detect objects, while the classification model is used to classify the detected objects.
The script saves the segmented objects as images in a folder.
"""

## INIT VIDEO CAPTURE
cap = cv2.VideoCapture("Testing_1.MOV")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

## INIT PARAMETERS
BLP_ID = 0
FRAME_ID = 0
model = YOLO('best.pt')

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    res = model.predict(im0, imgsz=640)
    # iterate detection results 
    for r in res:
        BLP_ID = 1
        img = np.copy(r.orig_img)
        img_name = Path(r.path).stem

        # iterate each object contour 
        for ci,c in enumerate(r):
            try:
                label = c.names[c.boxes.cls.tolist().pop()]

                b_mask = np.zeros(img.shape[:2], np.uint8)

                # Create contour mask 
                contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

                # Isolate object with transparent background (when saved as PNG)
                isolated = np.dstack([img, b_mask])

                # Detection crop (from either OPT1 or OPT2)
                x1, y1, x2, y2 = c.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
                iso_crop = isolated[y1:y2, x1:x2]
                cv2.imwrite(f"saved2/BLP{BLP_ID}_frame{FRAME_ID}.png", iso_crop)
                BLP_ID += 1
            except Exception:
                pass

        FRAME_ID += 1

cap.release()
cv2.destroyAllWindows()

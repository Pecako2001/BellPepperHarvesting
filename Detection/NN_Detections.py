import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import argparse
import os

# Argument parser
parser = argparse.ArgumentParser(description="Object Segmentation and Classification using YOLOv8")
parser.add_argument('--model_path', type=str, required=True, help='Path to the YOLOv8 model')
parser.add_argument('--video_path', type=str, required=True, help='Path to the input video file')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output images')
args = parser.parse_args()

# INIT VIDEO CAPTURE
cap = cv2.VideoCapture(args.video_path)
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# INIT PARAMETERS
BLP_ID = 0
FRAME_ID = 0
model = YOLO(args.model_path)

# Ensure output directory exists
os.makedirs(args.output_dir, exist_ok=True)

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
        for ci, c in enumerate(r):
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
                cv2.imwrite(os.path.join(args.output_dir, f"BLP{BLP_ID}_frame{FRAME_ID}.png"), iso_crop)
                BLP_ID += 1
            except Exception:
                pass

        FRAME_ID += 1

cap.release()
cv2.destroyAllWindows()

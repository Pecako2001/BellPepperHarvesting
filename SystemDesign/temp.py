import cv2
import numpy as np
from ultralytics import YOLO, solutions
from tqdm import tqdm
import argparse
import concurrent.futures

# Argument parser
parser = argparse.ArgumentParser(description="Bell Pepper Counting with YOLO")
parser.add_argument('--input_video', type=str, required=True, help='Path to the input video file')
parser.add_argument('--output_video', type=str, required=True, help='Path to save the output video file')
parser.add_argument('--seg_model', type=str, required=True, help='Path to the segmentation model')
parser.add_argument('--cls_model', type=str, required=True, help='Path to the classification model')
args = parser.parse_args()

# Load the YOLO models and suppress logging
model = YOLO(args.seg_model)
model.overrides['verbose'] = False

Clsmodel = YOLO(args.cls_model)
Clsmodel.overrides['verbose'] = False
names = model.model.names

cap = cv2.VideoCapture(args.input_video)
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

line_points = [(1080, 0), (1080, 3200)]  # line or region points
classes_to_count = [0]  # person and car classes for count

# Video writer
video_writer = cv2.VideoWriter(args.output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Init Object Counter
counter = solutions.ObjectCounter(
    view_img=False,
    reg_pts=line_points,
    classes_names=model.names,
    draw_tracks=False,
    line_thickness=3,
)

# Data structure to store classification results
classified_objects = {}
templist = []

ripe_count = 0
unripe_count = 0

# Function to classify a bounding box image
def classify_bbox(track_id, bbox_image):
    results = Clsmodel(bbox_image)
    class_label = results[0].probs.top1  # Extract class label from classification
    return track_id, class_label

# Initialize the progress bar
with tqdm(total=total_frames, desc="Processing frames") as pbar:
    with concurrent.futures.ThreadPoolExecutor(8) as executor:
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            # Perform segmentation
            tracks = model.track(im0, persist=True, show=False, classes=classes_to_count, conf=0.2)

            if tracks[0].masks is not None:
                clss = tracks[0].boxes.cls.cpu().tolist()
                masks = tracks[0].masks.data.cpu().numpy()
                track_ids = tracks[0].boxes.id.cpu().tolist()

                future_to_track_id = {}
                for i, mask in enumerate(masks):
                    track_id = track_ids[i]
                    
                    if track_id not in classified_objects:
                        # Convert the mask to a suitable image format for classification
                        mask_image = (mask * 255).astype('uint8')
                        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)

                        # Get bounding box coordinates for the mask
                        x1, y1, x2, y2 = map(int, tracks[0].boxes.xyxy[i])

                        # Extract the bounding box image
                        bbox_image = im0[y1:y2, x1:x2]

                        # Submit classification task to the executor
                        future = executor.submit(classify_bbox, track_id, bbox_image)
                        future_to_track_id[future] = track_id

                for future in concurrent.futures.as_completed(future_to_track_id):
                    track_id, class_label = future.result()
                    classified_objects[track_id] = class_label

                    if class_label == 0:
                        ripe_count += 1
                        label = "Ripe"
                        color = (0, 255, 0)  # Green
                    else:
                        unripe_count += 1
                        label = "Unripe"
                        color = (0, 0, 255)  # Red

                    # Get bounding box coordinates for the mask
                    x1, y1, x2, y2 = map(int, tracks[0].boxes.xyxy[track_ids.index(track_id)])

                    # Draw classification label above the detection
                    cv2.putText(im0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Update the counter and get the counts
            im0 = counter.start_counting(im0, tracks)
            temp = counter.count_ids
            #print(f"temp = {temp} and track_ids = {track_ids}") 
            for track_id in temp:
                if track_id not in templist:
                    templist.append(track_id)

            # Draw the modern menu at the top left corner
            menu_height = 90  # Increase the height of the menu
            menu_width = 450  # Increase the width of the menu
            menu_background = np.zeros((menu_height, menu_width, 3), dtype=np.uint8)
            cv2.rectangle(menu_background, (0, 0), (menu_width, menu_height), (50, 50, 50), -1)  # Background color
            cv2.putText(menu_background, "Bell Pepper Count", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(menu_background, f"Ripe: {ripe_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(menu_background, f"Unripe: {unripe_count}", (220, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            # Overlay the menu on the original frame
            im0[0:menu_height, 0:menu_width] = menu_background

            video_writer.write(im0)
            
            # Update the progress bar
            pbar.update(1)

cap.release()
video_writer.release()
cv2.destroyAllWindows()

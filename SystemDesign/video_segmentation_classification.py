# Import necessary libraries
import cv2
from ultralytics import YOLO
import numpy as np

"""
This script demonstrates how to perform object segmentation and classification on a video using YOLOv8.
The script loads a video file, segments objects in each frame, and classifies the segmented objects.
The segmentation model is used to detect objects, while the classification model is used to classify the detected objects.
The script displays the video frames with segmentation and classification results in real-time.
"""

# Load the YOLOv8 segmentation and classification models
segmentation_model = YOLO("best.pt")  # Replace with your segmentation model path
classification_model = YOLO("Classify/runs/classify/train/weights/best.pt")  # Replace with your classification model path

# Define colors for each class
segmentation_colors = {
    'Bell pepper': (0, 255, 0)  # Green
}
classification_colors = {
    'Unripe': (0, 0, 255),        # Red
    'Ripe': (0, 255, 255)         # Yellow
}

# Function to apply classification model to segmented objects
def classify_objects(image, boxes):
    """Classify the objects in the image using the classification model."""
    classified_objects = []
    for box in boxes:
        # Extract the bounding box coordinates and crop the object
        x1, y1, x2, y2 = map(int, box)
        cropped_img = image[y1:y2, x1:x2]

        # Run classification on the cropped object
        results = classification_model(cropped_img)
        if results and results[0].probs is not None:
            top1 = results[0].probs.top1
            top1_confidence = results[0].probs.top1conf.item()
            classification = results[0].names[top1]
            classified_objects.append((classification, (x1, y1, x2, y2), top1_confidence))

    return classified_objects

def draw_results(image, segmentation_results, classification_results):
    """Draw the segmentation and classification results on the image."""
    overlay = image.copy()

    # Draw only Bell pepper segmentation masks and their bounding boxes
    for r in segmentation_results[0].boxes:
        box = r.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, box)
        cls = int(r.cls.item())
        label = segmentation_model.names[cls]
        if label == 'Bell pepper':
            color = segmentation_colors['Bell pepper']
            conf = r.conf.item()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)  # Bold bounding box
            cv2.putText(overlay, f"{label} {conf:.2f}", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)  # Larger font size, bold

    # Draw classification results inside Bell pepper bounding boxes
    for class_name, box, confidence in classification_results:
        x1, y1, x2, y2 = map(int, box)
        color = classification_colors.get(class_name, (255, 255, 255))  # White as default color
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(overlay, label, (x1, y1 - 60), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)  # Larger font size, bold, with more spacing

    # Blend the original image with the overlay
    cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

    return image


# Initialize the video capture
video_path = "Testing_1.MOV"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Define the codec and create VideoWriter object
output_path = "output_video.mp4"  # Path to save the processed video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec used to compress the frames
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Set OpenCV fullscreen window
cv2.namedWindow("YOLOv8 Segmentation and Classification", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("YOLOv8 Segmentation and Classification", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Loop through the video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Segment objects in the current frame, focusing on Bell pepper (class 0)
    segmentation_results = segmentation_model(frame)
    segmentation_boxes = segmentation_results[0].boxes.xyxy.cpu().numpy()
    segmentation_classes = segmentation_results[0].boxes.cls.cpu().numpy()
    bell_pepper_boxes = segmentation_boxes[segmentation_classes == 0]

    # Classify each Bell pepper object
    classified_objects = classify_objects(frame, bell_pepper_boxes)

    # Draw both segmentation and classification results
    annotated_frame = draw_results(frame, segmentation_results, classified_objects)

    # Write the annotated frame to the output video
    out.write(annotated_frame)

    # Display the annotated frame in fullscreen
    cv2.imshow("YOLOv8 Segmentation and Classification", annotated_frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

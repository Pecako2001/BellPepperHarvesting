"""
This script demonstrates how to perform object segmentation and classification on a video using YOLOv8.
The script loads a video file, segments objects in each frame, and classifies the segmented objects.
The segmentation model is used to detect objects, while the classification model is used to classify the detected objects.
The script displays the video frames with segmentation and classification results in real-time.

Exampel usage: python python_Full_System_Video.py --video input_video.mp4 --ClassifyModel ClassBest.pt --SegModel SegBest.pt --output output_video.mp4 --show true
"""
# Import necessary libraries
import cv2
from ultralytics import YOLO
import numpy as np
import argparse
import matplotlib.pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser(description='YOLOv8 Segmentation and Classification on a Video')
    parser.add_argument('--video', required=True, help='Path to the input video')
    parser.add_argument('--ClassifyModel', required=True, help='Path to the classification model')
    parser.add_argument('--SegModel', required=True, help='Path to the segmentation model')
    parser.add_argument('--output', required=True, help='Path to save the output video')
    parser.add_argument('--show', action='store_true', help='Show video frames in real-time')
    return parser.parse_args()

# Function to apply classification model to segmented objects
def classify_objects(image, boxes, classification_model):
    classified_objects = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cropped_img = image[y1:y2, x1:x2]
        results = classification_model(cropped_img)
        if results and results[0].probs is not None:
            top1 = results[0].probs.top1
            top1_confidence = results[0].probs.top1conf.item()
            classification = results[0].names[top1]
            classified_objects.append((classification, (x1, y1, x2, y2), top1_confidence))
    return classified_objects

def draw_results(image, segmentation_results, classification_results):
    overlay = image.copy()
    height, width, _ = image.shape
    
    # Apply masks for all detected objects
    if segmentation_results[0].masks is not None:
        for mask, cls in zip(segmentation_results[0].masks.data.cpu().numpy(), segmentation_results[0].boxes.cls.cpu().numpy()):
            mask_resized = cv2.resize(mask.squeeze(), (width, height), interpolation=cv2.INTER_NEAREST)
            label = segmentation_model.names[int(cls)]
            color = segmentation_colors.get(label, (255, 255, 255))  # Default to white if class color is not defined
            image[mask_resized > 0.5] = image[mask_resized > 0.5] * 0.5 + np.array(color) * 0.5

    # Draw bounding boxes and labels for all detected objects
    for r in segmentation_results[0].boxes:
        box = r.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, box)
        cls = int(r.cls.item())
        label = segmentation_model.names[cls]
        color = segmentation_colors.get(label, (255, 255, 255))
        conf = r.conf.item()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
        cv2.putText(overlay, f"{label} {conf:.2f}", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
    
    # Draw classification results for Bell pepper objects
    for class_name, box, confidence in classification_results:
        x1, y1, x2, y2 = map(int, box)
        color = classification_colors.get(class_name, (255, 255, 255))
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(overlay, label, (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 4)

    cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
    return image

if __name__ == "__main__":
    args = parse_arguments()
    
    # Load the YOLOv8 segmentation and classification models
    segmentation_model = YOLO(args.SegModel)
    classification_model = YOLO(args.ClassifyModel)
    
    # Define colors for each class
    segmentation_colors = {
        'Bell pepper': (0, 255, 0),  # Green
        # Add more classes and their colors here if needed
    }
    classification_colors = {
        'Unripe': (0, 0, 255),        # Red
        'Ripe': (0, 255, 255)         # Yellow
    }
    
    # Initialize the video capture
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise ValueError("Error opening video stream or file")
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize the video writer
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
    if args.show:
        # Set up Matplotlib for interactive plotting
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 10))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Segment objects in the frame
        segmentation_results = segmentation_model(frame)
        if len(segmentation_results[0].boxes) == 0:
            print("No detections found in this frame.")
            annotated_frame = frame
        else:
            segmentation_boxes = segmentation_results[0].boxes.xyxy.cpu().numpy()
            segmentation_classes = segmentation_results[0].boxes.cls.cpu().numpy()
            bell_pepper_boxes = segmentation_boxes[segmentation_classes == 0]
            
            # Classify each Bell pepper object
            classified_objects = classify_objects(frame, bell_pepper_boxes, classification_model)
            
            # Draw both segmentation and classification results
            annotated_frame = draw_results(frame, segmentation_results, classified_objects)
        
        # Write the annotated frame to the output video
        out.write(annotated_frame)
        
        if args.show:
            # Display the annotated frame
            ax.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
            ax.axis('off')
            plt.draw()
            plt.pause(0.001)
            ax.clear()
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    if args.show:
        plt.ioff()
        plt.show()

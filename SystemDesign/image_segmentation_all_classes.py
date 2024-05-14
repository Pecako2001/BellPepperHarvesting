# Import necessary libraries
import cv2
from ultralytics import YOLO
import numpy as np
import argparse
import matplotlib.pyplot as plt

"""
This script demonstrates how to perform object segmentation and classification on an image using YOLOv8.
The script loads an image, segments objects in the image, classifies the segmented objects, and displays the results.
The segmentation model is used to detect objects, while the classification model is used to classify the detected objects.

Example usage: python python_Full_System_Photo.py --image blp_001.png --ClassifyModel ClassBest.pt --SegModel SegBest.pt
"""

def parse_arguments():
    parser = argparse.ArgumentParser(description='YOLOv8 Segmentation and Classification on an Image')
    parser.add_argument('--image', required=True, help='Path to the input image')
    parser.add_argument('--ClassifyModel', required=True, help='Path to the classification model')
    parser.add_argument('--SegModel', required=True, help='Path to the segmentation model')
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
    
    # Read the input image
    image = cv2.imread(args.image)
    if image is None:
        raise ValueError("Image not found or invalid image path")
    
    # Segment objects in the image
    segmentation_results = segmentation_model(image)
    if len(segmentation_results[0].boxes) == 0:
        print("No detections found.")
    else:
        segmentation_boxes = segmentation_results[0].boxes.xyxy.cpu().numpy()
        segmentation_classes = segmentation_results[0].boxes.cls.cpu().numpy()
        bell_pepper_boxes = segmentation_boxes[segmentation_classes == 0]
        
        # Classify each Bell pepper object
        classified_objects = classify_objects(image, bell_pepper_boxes, classification_model)
        
        # Draw both segmentation and classification results
        annotated_image = draw_results(image, segmentation_results, classified_objects)
        
        # Display the annotated image
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title('YOLOv8 Segmentation and Classification')
        plt.show()

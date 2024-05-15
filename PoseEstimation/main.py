import cv2
from ultralytics import YOLO
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

# Load the YOLOv8 segmentation model
model = YOLO('SegBest.pt')

# Load the image
image_path = 'TestImages/3D_BLP_006.jpg'
image = cv2.imread(image_path)

# Run YOLOv8 inference on the image
results = model(image, imgsz=320, iou=0.2, conf=0.20)

# Visualize the results on the image
annotated_image = results[0].plot()

# Display the annotated image
plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
plt.title('YOLOv8 Inference')
plt.axis('off')
plt.show()

# Process each detection result
for r in results:
    img = np.copy(r.orig_img)
    
    # Iterate each object contour
    for c in r:
        label = c.names[c.boxes.cls.tolist().pop()]

        b_mask = np.zeros(img.shape[:2], np.uint8)

        # Create contour mask
        contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
        _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

        # Find contours in the mask
        contours, _ = cv2.findContours(b_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Assuming the largest contour corresponds to the bell pepper
            contour = max(contours, key=cv2.contourArea)

            # Fit an ellipse to the contour
            ellipse = cv2.fitEllipse(contour)

            # Draw the ellipse on the original image
            image_with_ellipse = image.copy()
            cv2.ellipse(image_with_ellipse, ellipse, (0, 255, 0), 2)

            # Calculate the minimum enclosing rotated rectangle (bounding box)
            rotated_rect = cv2.minAreaRect(contour)
            box_points = cv2.boxPoints(rotated_rect)
            box_points = np.intp(box_points)

            # Draw the bounding box on the original image
            cv2.drawContours(image_with_ellipse, [box_points], 0, (0, 0, 255), 2)

            # Draw an arrow pointing to the peduncle of the bell pepper
            peduncle_point = (int(ellipse[0][0]), int(ellipse[0][1] - ellipse[1][1] / 2))
            cv2.arrowedLine(image_with_ellipse, peduncle_point, (peduncle_point[0], peduncle_point[1] - 50), (255, 0, 0), 2, tipLength=0.3)

            # Extract pose information
            center = rotated_rect[0]
            size = rotated_rect[1]
            angle = rotated_rect[2]

            # Display the result
            image_with_ellipse_rgb = cv2.cvtColor(image_with_ellipse, cv2.COLOR_BGR2RGB)
            plt.imshow(image_with_ellipse_rgb)
            plt.title('Pose Estimation of Green Bell Pepper')
            plt.axis('off')
            plt.show()

            # Print pose information
            print(f"Center: {center}")
            print(f"Size: {size}")
            print(f"Angle: {angle}")
else:
    print("No masks found in the image.")

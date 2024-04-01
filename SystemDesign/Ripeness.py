import cv2
import numpy as np

""" 
This file is used to calculate the ripeness of a bell pepper image.
It uses color thresholding in HSV color space to detect yellow and red colors.
The ripeness is determined based on the coverage of yellow and red colors in the image.
at least 80% coverage is considered ripe.
"""

def calculate_ripeness(image_path):
    # Load the image
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges for yellow and red in HSV
    # Note: You might need to adjust these ranges
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])
    red_lower1 = np.array([0, 100, 100])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 100, 100])
    red_upper2 = np.array([179, 255, 255])

    # Threshold the HSV image to get only yellow and red colors
    mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
    mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
    mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    # Combine masks
    combined_mask = cv2.bitwise_or(mask_yellow, mask_red)

    # Calculate coverage
    coverage = np.sum(combined_mask > 0) / (image.shape[0] * image.shape[1]) * 100

    # Determine ripeness based on coverage
    is_ripe = coverage >= 80

    return is_ripe, coverage

# Replace 'your_image_path.jpg' with the path to your bell pepper image
is_ripe, coverage = calculate_ripeness('ultralytics_crop/144.png')
print(f"Is the bell pepper ripe? {'Yes' if is_ripe else 'No'}")
print(f"Coverage: {coverage:.2f}%")

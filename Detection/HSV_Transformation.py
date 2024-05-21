import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse
import os

# Argument parser
parser = argparse.ArgumentParser(description="HSV Transformation and Segmentation")
parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output images')
parser.add_argument('--threshold', type=int, default=130, help='Threshold value for segmentation')
args = parser.parse_args()

# Load the image
image = cv2.imread(args.image_path)

# Convert to LAB color space
image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# Split the LAB image into its channels
L, A, B = cv2.split(image_lab)

# Save the L, A, and B channel images
os.makedirs(args.output_dir, exist_ok=True)
cv2.imwrite(os.path.join(args.output_dir, 'L_channel.png'), L)
cv2.imwrite(os.path.join(args.output_dir, 'A_channel.png'), A)
cv2.imwrite(os.path.join(args.output_dir, 'B_channel.png'), B)

# Display the LAB channels
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(L, cmap='gray')
plt.title('L Channel')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(A, cmap='gray')
plt.title('A Channel')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(B, cmap='gray')
plt.title('B Channel')
plt.axis('off')
plt.show()

# Function to adjust the thresholds
def segment_green_bell_peppers(A_channel, threshold_value):
    # Apply threshold to A channel to isolate green regions
    _, mask = cv2.threshold(A_channel, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # Apply morphological operations to remove noise and enhance the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_morph = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_morph = cv2.morphologyEx(mask_morph, cv2.MORPH_OPEN, kernel)
    
    # Apply the refined mask to the original image
    segmented = cv2.bitwise_and(image, image, mask=mask_morph)
    
    return segmented, mask_morph

# Adjust the threshold value to get a better separation inside the green colors
segmented_image, final_mask = segment_green_bell_peppers(A, args.threshold)

# Save the final segmented image and mask
cv2.imwrite(os.path.join(args.output_dir, 'final_segmented_image.png'), segmented_image)
cv2.imwrite(os.path.join(args.output_dir, 'final_mask.png'), final_mask)

# Convert the segmented image to RGB for displaying
segmented_image_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)

# Display the result
plt.imshow(segmented_image_rgb)
plt.title(f'Segmented Green Bell Peppers (Threshold: {args.threshold})')
plt.axis('off')
plt.show()

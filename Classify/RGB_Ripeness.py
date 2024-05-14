import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
This script demonstrates how to classify the ripeness of fruits in images using OpenCV and NumPy.
The script loads images of fruits, calculates the ripeness based on color, and displays the results.
"""

def calculate_ripeness(image_path):
    """Calculate the ripeness of a fruit in an image."""
    # Load the image
    image = cv2.imread(image_path)
    gamma_corrected = adjust_gamma(image, gamma=5) # You may need to experiment with the gamma value
    hsv = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2HSV)

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Split into channels
    h, s, v = cv2.split(hsv)

    # Apply CLAHE to the V channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_clahe = clahe.apply(v)

    # Merge the channels back
    hsv_clahe = cv2.merge([h, s, v_clahe])

    # Define color ranges for yellow and red in HSV
    yellow_lower = np.array([20, 100, 25])
    yellow_upper = np.array([30, 255, 255])
    red_lower1 = np.array([0, 25, 25])   # Red hue range lower part
    red_upper1 = np.array([10, 255, 255])  # Slightly reduce the upper hue to avoid picking up orange tones
    red_lower2 = np.array([170, 25, 25])  # Red hue range upper part
    red_upper2 = np.array([180, 255, 255]) 


    # Threshold the CLAHE-modified HSV image to get only yellow and red colors
    mask_yellow = cv2.inRange(hsv_clahe, yellow_lower, yellow_upper)
    mask_red1 = cv2.inRange(hsv_clahe, red_lower1, red_upper1)
    mask_red2 = cv2.inRange(hsv_clahe, red_lower2, red_upper2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Combine masks
    combined_mask = cv2.bitwise_or(mask_yellow, mask_red)

    # Calculate coverage
    coverage = np.sum(combined_mask > 0) / (image.shape[0] * image.shape[1]) * 100

    # Determine ripeness based on coverage
    is_ripe = coverage >= 25  # You may need to adjust this threshold

    return is_ripe, coverage

def adjust_gamma(image, gamma=1.0):
    """Adjust the gamma of an image."""
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def process_directory(directory):
    """Process all images in a directory and return the results."""
    images = []
    predictions = []
    for filename in tqdm(os.listdir(directory), desc="Processing images"):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory, filename)
            is_ripe, coverage = calculate_ripeness(image_path)
            img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)  # Convert color for matplotlib
            images.append((img, is_ripe, coverage))
    return images

def plot_and_save_results(base_dir, save_dir, fig_size=(15, 15), font_size=14):
    """Plot and save the results of the classification."""
    ripe_images = process_directory(os.path.join(base_dir, 'ripe'))
    unripe_images = process_directory(os.path.join(base_dir, 'unripe'))

    total_images = len(ripe_images) + len(unripe_images)
    cols, rows = 5, 5
    max_images_per_batch = cols * rows

    # Make sure the directory exists where we will save the individual images
    os.makedirs(save_dir, exist_ok=True)
    """Uncomment this block to save the images in batches."""
    # for batch_index in range(0, total_images, max_images_per_batch):
    #     fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=fig_size)
    #     axes = axes.ravel()

    #     # Get the current batch of images
    #     current_batch = (ripe_images + unripe_images)[batch_index:batch_index + max_images_per_batch]

    #     for i in range(max_images_per_batch):
    #         if i < len(current_batch):
    #             img, is_ripe, coverage = current_batch[i]
    #             axes[i].imshow(img)
    #             ripe_status = 'Ripe' if is_ripe else 'Unripe'
    #             title = f'{ripe_status}\nCoverage: {coverage:.2f}%'
    #             axes[i].set_title(title, fontsize=font_size)
    #         axes[i].axis('off')

    #     plt.subplots_adjust(hspace=0.5, wspace=0.3)
    #     plt.tight_layout()

    #     # Save this batch's figure
    #     fig_filename = f'classification_results_{batch_index // max_images_per_batch + 1}.png'
    #     plt.savefig(os.path.join(save_dir, fig_filename), bbox_inches='tight')
    #     plt.close(fig)

    # Calculate success rates
    ripe_success = sum(pred[1] for pred in ripe_images) / len(ripe_images) if ripe_images else 0
    unripe_success = sum(not pred[1] for pred in unripe_images) / len(unripe_images) if unripe_images else 0
    print(f"Success rate for ripe detection: {ripe_success * 100:.2f}%")
    print(f"Success rate for unripe detection: {unripe_success * 100:.2f}%")

# Example usage
plot_and_save_results('Classify/RGB_Classify_Full', 'runs/rgb_classify_set2')
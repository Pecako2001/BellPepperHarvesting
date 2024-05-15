import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

"""
This script demonstrates how to classify the ripeness of fruits in images using OpenCV and NumPy.
The script loads images of fruits, calculates the ripeness based on color, and displays the results.

Example usage: python RGB_Ripeness.py base_dir save_dir --fig_size 15 15 --font_size 14
"""
def parse_arguments():
    parser = argparse.ArgumentParser(description="Classify the ripeness of fruits in images.")
    parser.add_argument('--input_dir', type=str, help="Base directory containing 'ripe' and 'unripe' subdirectories.")
    parser.add_argument('--output_dir', type=str, help="Directory where the output images and results will be saved.")
    return parser.parse_args()

def calculate_ripeness(image_path):
    """Calculate the ripeness of a fruit in an image."""
    # Load the image
    image = cv2.imread(image_path)
    gamma_corrected = adjust_gamma(image, gamma=5)
    hsv = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2HSV)

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
    red_lower1 = np.array([0, 25, 25])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 25, 25])
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

def process_directory(directory, actual_status):
    """Process all images in a directory and return the results."""
    images = []
    for filename in tqdm(os.listdir(directory), desc="Processing images"):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory, filename)
            is_ripe, coverage = calculate_ripeness(image_path)
            img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            images.append((img, is_ripe, coverage, filename, actual_status))
    return images

def plot_and_save_results(base_dir, save_dir, fig_size=(15, 15), font_size=14):
    """Plot and save the results of the classification."""
    ripe_images = process_directory(os.path.join(base_dir, 'ripe'), 'Ripe')
    unripe_images = process_directory(os.path.join(base_dir, 'unripe'), 'Unripe')

    total_images = len(ripe_images) + len(unripe_images)
    cols, rows = 5, 5
    max_images_per_batch = cols * rows

    os.makedirs(save_dir, exist_ok=True)
    for batch_index in range(0, total_images, max_images_per_batch):
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=fig_size)
        axes = axes.ravel()

        current_batch = (ripe_images + unripe_images)[batch_index:batch_index + max_images_per_batch]

        for i in range(max_images_per_batch):
            if i < len(current_batch):
                img, is_ripe, coverage, filename, actual_status = current_batch[i]
                axes[i].imshow(img)
                detected_status = 'Ripe' if is_ripe else 'Unripe'
                title = f'Actual: {actual_status}\nDetected: {detected_status}\nCoverage: {coverage:.2f}%'
                axes[i].set_title(title, fontsize=font_size)
            axes[i].axis('off')

        plt.subplots_adjust(hspace=0.5, wspace=0.3)
        plt.tight_layout()

        fig_filename = f'RGB_classification_results_{batch_index // max_images_per_batch + 1}.png'
        plt.savefig(os.path.join(save_dir, fig_filename), bbox_inches='tight')
        plt.close(fig)

    ripe_success = sum(pred[1] for pred in ripe_images) / len(ripe_images) if ripe_images else 0
    unripe_success = sum(not pred[1] for pred in unripe_images) / len(unripe_images) if unripe_images else 0
    print(f"Success rate for ripe detection: {ripe_success * 100:.2f}%")
    print(f"Success rate for unripe detection: {unripe_success * 100:.2f}%")

if __name__ == "__main__":
    args = parse_arguments()

    plot_and_save_results(args.input_dir, args.output_dir)

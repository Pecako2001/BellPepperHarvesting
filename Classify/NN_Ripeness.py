import cv2
from ultralytics import YOLO
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description='Classification of Images')
    parser.add_argument('--input_dir', required=True, help='Path to the base directory containing "ripe" and "unripe" subdirectories')
    parser.add_argument('--ClassifyModel', required=True, help='Path to the classification model')
    parser.add_argument('--output_dir', required=True, help='Directory to save the output results')
    return parser.parse_args()

def classify_image(image_path, classification_model):
    image = cv2.imread(image_path)
    results = classification_model(image)
    if results and results[0].probs is not None:
        top1 = results[0].probs.top1
        top1_confidence = results[0].probs.top1conf.item()
        classification = results[0].names[top1]
        return classification, top1_confidence
    return None, 0

def process_directory(directory, actual_status, classification_model, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    images = []
    for filename in tqdm(os.listdir(directory), desc=f"Processing {actual_status} images"):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory, filename)
            classification, confidence = classify_image(image_path, classification_model)
            image = cv2.imread(image_path)
            images.append((image, classification, confidence, filename, actual_status))

    return images

def plot_and_save_results(base_dir, output_dir, fig_size=(15, 15), font_size=14):
    ripe_images = process_directory(os.path.join(base_dir, 'ripe'), 'Ripe', classification_model, output_dir)
    unripe_images = process_directory(os.path.join(base_dir, 'unripe'), 'Unripe', classification_model, output_dir)

    total_images = len(ripe_images) + len(unripe_images)
    cols, rows = 5, 5
    max_images_per_batch = cols * rows

    for batch_index in range(0, total_images, max_images_per_batch):
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=fig_size)
        axes = axes.ravel()

        current_batch = (ripe_images + unripe_images)[batch_index:batch_index + max_images_per_batch]

        for i in range(max_images_per_batch):
            if i < len(current_batch):
                img, classification, confidence, filename, actual_status = current_batch[i]
                axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                detected_status = classification
                title = f'Actual: {actual_status}\nDetected: {detected_status}\nConfidence: {confidence:.2f}%'
                axes[i].set_title(title, fontsize=font_size)
            axes[i].axis('off')

        plt.tight_layout()
        batch_output_path = os.path.join(output_dir, f'NN_classification_results_{batch_index // max_images_per_batch + 1}.png')
        plt.savefig(batch_output_path, bbox_inches='tight')
        plt.close(fig)

if __name__ == "__main__":
    args = parse_arguments()
    
    # Load the YOLOv8 classification model
    classification_model = YOLO(args.ClassifyModel)
    
    # Plot and save results
    plot_and_save_results(args.input_dir, args.output_dir)

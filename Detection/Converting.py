import cv2
import numpy as np
import os
import albumentations as A

class SegmentationProcessorAndAugmentor:
    def __init__(self, input_dir, mask_dir, output_dir):
        self.input_dir = input_dir
        self.mask_dir = mask_dir
        self.output_dir = output_dir
        # Define the augmentations
        self.augmentations = A.Compose([
            A.InvertImg(p=1),  # Inverts the input image colors.
            A.VerticalFlip(p=1),  # Flips the input image vertically.
            A.HorizontalFlip(p=1),  # Flips the input image horizontally.
            A.Blur(blur_limit=7, p=1),  # Applies Blur to the input image.
            A.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=1),  # Randomly drops channels from the input image.
            A.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=1),  # Randomly drops channels from the input image.
            A.ToSepia(p=1),  # Applies Sepia filter to the input image.
            A.ChannelShuffle(p=1),  # Randomly shuffles the channels of the input image.
        ])
        self.augmentations_with_suffixes = [
            (None, ''),  # No augmentation for the original image
            (A.InvertImg(p=1.0), '_inverted'),
            (A.VerticalFlip(p=1.0), '_vflipped'),
            (A.HorizontalFlip(p=1.0), '_hflipped'),
            (A.Blur(blur_limit=7, p=1.0), '_blurred'),
            (A.ChannelDropout(p=1.0), '_channeldropped'),
            (A.ChannelDropout(p=1.0), '_channeldropp'),
            (A.ToSepia(p=1.0), '_sepia'),
            (A.ChannelShuffle(p=1.0), '_ChannelShuffle'),
        ]

    def process_segmentation_file(self, txt_path, img_path, augmentation=None, aug_suffix=""):
        original_img = cv2.imread(img_path)
        height, width = original_img.shape[:2]

        bw_img = np.zeros((height, width), dtype=np.uint8)  # Assuming mask is single-channel

        # Read annotation and convert to mask
        with open(txt_path, 'r') as file:
            for line in file.readlines():
                parts = list(map(float, line.split()))
                class_id = int(parts[0])
                x_center, y_center, w, h = parts[1:5]
                x_center, y_center, w, h = x_center * width, y_center * height, w * width, h * height
                x_min, y_min = int(x_center - w / 2), int(y_center - h / 2)

                cv2.rectangle(bw_img, (x_min, y_min), (x_min + int(w), y_min + int(h)), (255), -1)

        if augmentation:
            augmented = augmentation(image=original_img, mask=bw_img)
            augmented_image = augmented['image']
            augmented_mask = augmented['mask']
        else:
            augmented_image, augmented_mask = original_img, bw_img

        base_filename = os.path.splitext(os.path.basename(img_path))[0]

        # Save the augmented image
        aug_img_path = os.path.join(self.output_dir, f"{base_filename}{aug_suffix}.png")
        cv2.imwrite(aug_img_path, augmented_image)
        height, width = augmented_image.shape[:2]

        # Convert the augmented mask back to YOLO format and save
        aug_txt_path = os.path.join(self.output_dir, f"{base_filename}{aug_suffix}.txt")        
        contours, _ = cv2.findContours(augmented_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        with open(aug_txt_path, 'w') as f:
            for cnt in contours:
                # Simplify contour to reduce the number of points
                epsilon = 0.001 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                # Normalize points and prepare for saving
                normalized_points = [(point[0][0] / width, point[0][1] / height) for point in approx]
                
                # Writing points to file
                f.write(f"{class_id} " + " ".join(f"{x} {y}" for x, y in normalized_points) + "\n")


    def process_all_files(self):
        for filename in os.listdir(self.input_dir):
            if filename.endswith('.png'):  # Adjust based on your input image format
                base_name = os.path.splitext(filename)[0]
                img_path = os.path.join(self.input_dir, filename)
                txt_path = os.path.join(self.input_dir, f"{base_name}.txt")

                if os.path.exists(txt_path):
                    # Iterate over each specified augmentation and its corresponding suffix
                    for augmentation, suffix in self.augmentations_with_suffixes:
                        # If no augmentation is specified, pass None, else pass the augmentation
                        self.process_segmentation_file(txt_path, img_path, augmentation, suffix)
                else:
                    print(f"Annotation file does not exist for {filename}")


if __name__ == "__main__":
    input_dir = "data"
    mask_dir = "data/masks"
    output_dir = "data/augmented"
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    processor = SegmentationProcessorAndAugmentor(input_dir, mask_dir, output_dir)
    processor.process_all_files()

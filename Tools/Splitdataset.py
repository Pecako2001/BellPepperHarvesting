import os
import shutil
from random import shuffle

"""
Tool to split a dataset into training and validation sets.
The script takes a source folder containing the dataset and splits it into two folders: train and val.
The train_ratio parameter specifies the ratio of files to move to the training set.
"""

def split_data(source_folder, train_folder, val_folder, train_ratio):
    # Create train and validation folders if they don't exist
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    # Get all files from the source folder
    files = [file for file in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, file))]
    shuffle(files)  # Randomize file order

    # Calculate split index
    split_index = int(len(files) * train_ratio)

    # Split files into train and validation sets
    train_files = files[:split_index]
    val_files = files[split_index:]

    # Move files to the corresponding folders
    for file in train_files:
        shutil.copy(os.path.join(source_folder, file), os.path.join(train_folder, file))
    for file in val_files:
        shutil.copy(os.path.join(source_folder, file), os.path.join(val_folder, file))

# Example usage:
source_dir = 'Classify/RGB_Classify_Dataset4/unripe'  # Replace with the path to your source directory
train_dir = 'Classify/NN_Classify_Dataset/train/unripe'    # Replace with the path to your train directory
val_dir = 'Classify/NN_Classify_Dataset/val/unripe'        # Replace with the path to your validation directory
train_ratio=0.91
split_data(source_dir, train_dir, val_dir, train_ratio)

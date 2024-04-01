import cv2
import os
import shutil

# Define the source folder where the images are stored
source_folder = 'ultralytics_crop'

# Define the destination folders for 'ripe' and 'unripe' classifications
ripe_folder = 'Classify/ripe'
unripe_folder = 'Classify/unripe'
delete_folder = 'Classify/delete'

# Create the 'ripe' and 'unripe' folders if they do not exist
os.makedirs(ripe_folder, exist_ok=True)
os.makedirs(unripe_folder, exist_ok=True)
os.makedirs(delete_folder, exist_ok=True)
# Get a list of images in the source folder
images = [f for f in os.listdir(source_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Loop through each image
for image_name in images:
    # Construct the full path to the image
    image_path = os.path.join(source_folder, image_name)
    
    # Read and display the image
    img = cv2.imread(image_path)
    cv2.imshow('Image', img)
    
    # Wait for the user to press a key
    key = cv2.waitKey(0) & 0xFF
    
    # If 'y' is pressed, move the image to the 'ripe' folder
    if key == ord('y'):
        shutil.move(image_path, os.path.join(ripe_folder, image_name))
        print(f"Moved {image_name} to 'ripe'")
    
    # If 'n' is pressed, move the image to the 'unripe' folder
    elif key == ord('n'):
        shutil.move(image_path, os.path.join(unripe_folder, image_name))
        print(f"Moved {image_name} to 'unripe'")

    # If 'd' is pressed, move the image to the 'delete' folder
    if key == ord('d'):
        shutil.move(image_path, os.path.join(delete_folder, image_name))
        print(f"Moved {image_name} to 'delete'")
    # Close the image window
    cv2.destroyAllWindows()

print("Processing completed.")

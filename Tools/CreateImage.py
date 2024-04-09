import cv2
import numpy as np

# Read two images
image1 = cv2.imread('00de04a9913eaa3cdf03210f215bbb1a.png')
image2 = cv2.imread('frame1450output.jpg')

# Resize images to the same height if they differ
height1, width1 = image1.shape[:2]
height2, width2 = image2.shape[:2]

# Find the maximum height of the two images
max_height = max(height1, height2)

# Scale images to the same height if they are not already
if height1 != max_height:
    scale_ratio = max_height / height1
    image1 = cv2.resize(image1, None, fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_LINEAR)
elif height2 != max_height:
    scale_ratio = max_height / height2
    image2 = cv2.resize(image2, None, fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_LINEAR)

# Stack images horizontally
hstacked_images = np.hstack((image1, image2))

# Set text properties
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 8 # Adjust the scale according to your image size
font_color = (255, 255, 255)  # White color for the text
thickness = 8 # Thickness of the text
border_thickness = 40  # Thickness for the black border
text1 = "Dataset image"
text2 = "Glass house image"

# Calculate text width & height to center the text
(text_width1, text_height1), _ = cv2.getTextSize(text1, font, font_scale, thickness)
(text_width2, text_height2), _ = cv2.getTextSize(text2, font, font_scale, thickness)

# Calculate the X position for text to be centered on each image
x1 = int((image1.shape[1] - text_width1) / 2)
x2 = int(image1.shape[1] + (image2.shape[1] - text_width2) / 2)

# Position the Y for the text to be on the top portion of each image
y_offset = 250
y = y_offset

# Draw the text with a black border
cv2.putText(hstacked_images, text1, (x1, y), font, font_scale, (0, 0, 0), border_thickness)
cv2.putText(hstacked_images, text2, (x2, y), font, font_scale, (0, 0, 0), border_thickness)

# Then draw the text again with the original settings on top
cv2.putText(hstacked_images, text1, (x1, y), font, font_scale, font_color, thickness)
cv2.putText(hstacked_images, text2, (x2, y), font, font_scale, font_color, thickness)

# Save the final image with text overlaid
cv2.imwrite('Combined_with_text_border.png', hstacked_images)
cv2.destroyAllWindows()

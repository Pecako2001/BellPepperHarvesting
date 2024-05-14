import cv2
import numpy as np

"""
Tool to resize a video while maintaining the aspect ratio.
This script reads a video file, resizes each frame to the specified dimensions while maintaining the aspect ratio,
and writes the resized video to a new file.
"""

def resize_video_with_aspect_ratio(input_video_path, output_video_path, width=1920, height=1080):
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    
    # Get the original video's width, height, and FPS
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate the scaling factors while maintaining aspect ratio
    scale_w = width / original_width
    scale_h = height / original_height
    scale = min(scale_w, scale_h)
    
    # Calculate new dimensions
    new_w = int(original_width * scale)
    new_h = int(original_height * scale)
    
    # Create a VideoWriter object to write the resized video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You may change 'mp4v' to 'XVID' if you face issues with codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize the frame
        resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create a black image and center the resized frame within it
        result_frame = np.zeros((height, width, 3), np.uint8)
        y_offset = (height - new_h) // 2
        x_offset = (width - new_w) // 2
        result_frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame
        
        # Write the frame into the output video
        out.write(result_frame)
    
    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage
input_video_path = 'test1.mp4'
output_video_path = 'Scaled_Test1.mp4'
resize_video_with_aspect_ratio(input_video_path, output_video_path)

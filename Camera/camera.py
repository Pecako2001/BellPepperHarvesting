from pypylon import pylon
import cv2

# Initialize camera and grab a single frame to determine the size
tlf = pylon.TlFactory.GetInstance()
cam = pylon.InstantCamera(tlf.CreateFirstDevice())
cam.Open()
cam.StartGrabbing()

with cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException) as result:
    if result.GrabSucceeded():
        # Convert to a PylonImage
        img = pylon.PylonImage()
        img.AttachGrabResultBuffer(result)
        
        # Get image dimensions
        width, height = img.GetWidth(), img.GetHeight()

cam.StopGrabbing()

# Video writer initialization
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 20  # Adjust FPS to your needs
video_filename = 'output_video.avi'
out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

# Start grabbing and writing frames
cam.StartGrabbing()
while True:
    with cam.RetrieveResult(2000) as result:
        if result.GrabSucceeded():
            # Convert to a PylonImage
            img.AttachGrabResultBuffer(result)

            # Convert PylonImage to OpenCV image format
            image_array = img.GetArray()

            # Write the frame
            out.write(image_array)
            cv2.imshow(image_array)
            img.Release()
        else:
            # End of stream or error; break from loop
            break

cam.StopGrabbing()
cam.Close()
out.release()

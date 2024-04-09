import cv2
import os
import time
import pypylon.pylon as py

icam = py.InstantCamera(py.TlFactory.GetInstance().CreateFirstDevice())
icam.Open()
cv2.namedWindow("1",cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow("1", 600,600)

# Initialize camera
#cap = cv2.VideoCapture(0)

ses = False

starting_time = time.time()
frame_id = 0
font = cv2.FONT_HERSHEY_PLAIN

while True:
    # Get frames
    # ret, frame = cap.read()

    img = icam.GrabOne(4000)
    img = img.Array
    frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    frame_id += 1

    # Object Detection
    # TODO

    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 4, (0, 0, 0), 3)
    imS = cv2.resize(frame, (960, 540))     
    cv2.imshow("Frame", imS)
    key = cv2.waitKey(1)
    if key == 27:
        break
cv2.destroyAllWindows()

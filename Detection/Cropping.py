from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import cv2
import os

model = YOLO("best.pt")
names = model.names

cap = cv2.VideoCapture("Scaled_Test1.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

crop_dir_name = "ultralytics_crop"
if not os.path.exists(crop_dir_name):
    os.mkdir(crop_dir_name)

idx = 0
frame_counter = 0  # Initialize a frame counter

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # Increment frame counter
    frame_counter += 1

    # Process every 100th frame
    if frame_counter % 1 == 0:
        results = model.predict(im0, show=False, conf=0.4)
        boxes = results[0].boxes.xyxy.cpu().tolist()
        clss = results[0].boxes.cls.cpu().tolist()

        if boxes is not None:
            for box, cls in zip(boxes, clss):
                idx += 1

                crop_obj = im0[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                cv2.imwrite(os.path.join(crop_dir_name, str(idx)+".png"), crop_obj)

    #cv2.imshow("ultralytics", im0)
    #video_writer.write(im0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
#video_writer.release()
cv2.destroyAllWindows()

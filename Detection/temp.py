import supervision as sv
from ultralytics import YOLO
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Define the paths for source and target video files and the model
source_path = "Testing_1.MOV"
target_path = "FullOutput.mp4"
model_path = "SegBest.pt"
classify_model_path = "ClassBest.pt"

# Load the video information and frame generator
video_info = sv.VideoInfo.from_video_path(video_path=source_path)
frame_generator = sv.get_video_frames_generator(source_path=source_path)

# Initialize the model, tracker, and annotators
model = YOLO(model_path)
classify_model = YOLO(classify_model_path)
tracker = sv.ByteTrack(frame_rate=video_info.fps)
mask_annotator = sv.MaskAnnotator()
label_annotator = sv.LabelAnnotator()

# Suppress logging
model.overrides['verbose'] = False
classify_model.overrides['verbose'] = False

# Initialize object counts
ripe_count = 0
unripe_count = 0

# Callback function to process each frame
def process_frame(frame: np.ndarray, index: int) -> np.ndarray:
    global ripe_count, unripe_count
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    if len(detections) > 0:  # Ensure there are detections before updating tracker
        detections = tracker.update_with_detections(detections)

        labels = []
        for tracker_id, box in zip(detections.tracker_id, detections.xyxy):
            # Extract the object for classification
            x1, y1, x2, y2 = map(int, box)
            object_img = frame[y1:y2, x1:x2]

            # Classify the object as ripe or unripe
            classify_results = classify_model(object_img)
            classify_label = classify_results[0].names[0]

            if classify_label == 'ripe':
                ripe_count += 1
            else:
                unripe_count += 1

            labels.append(f"#{tracker_id} {classify_label}")

    else:
        labels = []

    annotated_frame = mask_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    # Add object counts to the frame
    cv2.putText(annotated_frame, f'Ripe Count: {ripe_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(annotated_frame, f'Unripe Count: {unripe_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    return annotated_frame

# Process the video and write the annotated frames to the target file
def main():
    with sv.VideoSink(target_path, video_info=video_info) as sink:
        with ThreadPoolExecutor(max_workers=8) as executor:
            total_frames = int(video_info.total_frames)  # Total number of frames to process
            pbar = tqdm(total=total_frames, desc="Processing frames", unit="frame")

            for index, frame in enumerate(frame_generator):
                future = executor.submit(process_frame, frame, index)
                annotated_frame = future.result()
                sink.write_frame(annotated_frame)
                pbar.update(1)

                # Explicitly release memory
                del future
                del frame
                del annotated_frame

            pbar.close()

if __name__ == '__main__':
    main()

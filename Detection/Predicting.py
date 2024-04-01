import supervision as sv
import numpy as np
from ultralytics import YOLO
import torch

VIDEO_PATH = 'Scaled_Test1.mp4'
# Ensure CUDA is available and select it as the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = YOLO('best.pt')

video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)

def process_frame(frame: np.ndarray, _) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    bounding_box_annotator = sv.MaskAnnotator()
    label_annotator = sv.LabelAnnotator()

    labels = [
        model.model.names[class_id]
        for class_id
        in detections.class_id
    ]

    annotated_image = bounding_box_annotator.annotate(
        scene=frame, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels)

    return annotated_image

sv.process_video(source_path=VIDEO_PATH, target_path=f"result_2.mp4", callback=process_frame)

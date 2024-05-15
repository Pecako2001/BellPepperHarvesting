
# YOLOv8 Segmentation and Classification

This repository contains scripts to perform object segmentation and classification on images and videos using YOLOv8. The segmentation model is used to detect objects, while the classification model is used to classify the detected objects.

## Scripts

### video_segmentation_classification.py

This script demonstrates how to perform object segmentation and classification on a video using YOLOv8. The script loads a video file, segments objects in each frame, and classifies the segmented objects. The segmentation model is used to detect objects, while the classification model is used to classify the detected objects. The script displays the video frames with segmentation and classification results in real-time.

#### Usage

```sh
python video_segmentation_bell_pepper.py --video input_video.mp4 --ClassifyModel ClassBest.pt --SegModel SegBest.pt --output output_video.mp4 --show
```

#### Arguments

- `--video`: Path to the input video file.
- `--ClassifyModel`: Path to the classification model.
- `--SegModel`: Path to the segmentation model.
- `--output`: Path to save the output video file.
- `--show`: Flag to display the video frames in real-time.

### image_segmentation_all_classes.py

This script demonstrates how to perform object segmentation and classification on all classes in an image using YOLOv8. The script loads an image file, segments objects, and classifies the segmented objects. The segmentation model is used to detect objects, while the classification model is used to classify the detected objects.

#### Usage

```sh
python image_segmentation_all_classes.py --image input_image.png --ClassifyModel ClassBest.pt --SegModel SegBest.pt
```

#### Arguments

- `--image`: Path to the input image file.
- `--ClassifyModel`: Path to the classification model.
- `--SegModel`: Path to the segmentation model.

### image_segmentation_bell_pepper.py

This script demonstrates how to perform object segmentation and classification on a specific class (Bell pepper) in an image using YOLOv8. The script loads an image file, segments objects, and classifies the segmented objects that belong to the Bell pepper class. The segmentation model is used to detect objects, while the classification model is used to classify the detected objects.

#### Usage

```sh
python image_segmentation_bell_pepper.py --image input_image.png --ClassifyModel ClassBest.pt --SegModel SegBest.pt
```

#### Arguments

- `--image`: Path to the input image file.
- `--ClassifyModel`: Path to the classification model.
- `--SegModel`: Path to the segmentation model.

### video_segmentation_classification_test.py

This script demonstrates how to perform object segmentation and classification on a test set using YOLOv8. The script loads a video file, segments objects in each frame, and classifies the segmented objects. The segmentation model is used to detect objects, while the classification model is used to classify the detected objects. The script displays the video frames with segmentation and classification results in real-time.

#### Usage

```sh
python video_segmentation_all_classes.py --video input_video.mp4 --ClassifyModel ClassBest.pt --SegModel SegBest.pt --output output_video.mp4 --show
```

#### Arguments

- `--video`: Path to the input video file.
- `--ClassifyModel`: Path to the classification model.
- `--SegModel`: Path to the segmentation model.
- `--output`: Path to save the output video file.
- `--show`: Flag to display the video frames in real-time.

## Requirements

- Python 3.x
- OpenCV
- Matplotlib
- NumPy
- YOLOv8

## Installation

1. Clone this repository.
2. Install the required dependencies:

```sh
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes.

## License

This project is licensed under the MIT License.
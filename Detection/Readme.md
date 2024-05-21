For the detections of the systems there have been tested several models to look at the performance of these models

| Model       | Box Precision | Accuracy | mAP50 |
|-------------|---------------|----------|-------|
| YOLOV5-seg  | 62.8%         | 50.0%    |  -    |
| YOLOV8-seg  | 63.0%         | 51.5%    |  -    |
| YOLOV9-seg  | 54.1%         | 53.7%    |  -    |
| GELAN-seg   | 71.6%         | 53.2%    |  -    |

example script :
```sh
python yolov8_segmentation_video.py --input input_video.mp4 --output output_video.mp4 --model yolov8-seg.pt
```
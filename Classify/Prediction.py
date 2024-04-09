from ultralytics import YOLO

class Classify():
    def __init__(self):
        self.model = YOLO('Classify/runs/classify/train2/weights/best.pt')
        self.classes = self.model.names

    def get_ripeness(self, frame):
        results = self.model.predict(frame)
        for result in results:
            return result.probs.top1
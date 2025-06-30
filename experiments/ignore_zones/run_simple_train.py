from ultralytics import YOLO


model = YOLO(model='yolov8n.pt', task='detect')

model.train(
    data='/Users/tilwidmann/Work/AiFish/datasets/ignore_zones/dataset.yaml',
    epochs=100,
    ignore_zone_cls=1,
    device='mps',
)
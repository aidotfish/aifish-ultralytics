from ultralytics import YOLO


model = YOLO(model='yolov8n.pt', task='detect')

model.train(
    data='/Users/tilwidmann/Work/AiFish/datasets/new_england_alg_test/dataset.yaml',
    epochs=2,
    classes=[2],
    ignore_zone_cls=[0]
)
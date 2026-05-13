from ultralytics import YOLO

model = YOLO("yolo26n.pt")

results = model.train(data="coco8.yaml", epochs=100, imgsz=640)


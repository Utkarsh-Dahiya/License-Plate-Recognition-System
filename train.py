from ultralytics import YOLO

# Load pretrained YOLO11 nano model
model = YOLO("yolo11n.pt")

# Train the model
model.train(
    data="yolo_dataset/data.yaml",
    epochs=30,
    imgsz=640,
    batch=4,
    project="models",
    name="license_plate_detector"
)
from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('yolo11n.pt')
    model.train(data='data/data.yaml', epochs=400, imgsz=640, batch=64, workers=4)
    
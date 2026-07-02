from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('best.pt')
    
    results = model(r"data\test\images\747_jpg.rf.6b8e87c6f8858ceb19dbf66d4d9d7e1e.jpg")
    
    results[0].show()
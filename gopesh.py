from ultralytics import YOLO

model = YOLO('best_openvino_model')
results = model('tcp://127.0.0.1:8888', stream=True)

for result in results:
    print(result.boxes, result.probs)
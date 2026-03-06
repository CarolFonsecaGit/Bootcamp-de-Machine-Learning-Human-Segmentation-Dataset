from ultralytics import YOLO

# Carrega modelo pré-treinado de segmentação
model = YOLO("runs/segment/train/weights/last.pt")  


# Treinar a partir do modelo pré-treinado
model.train(resume=True)
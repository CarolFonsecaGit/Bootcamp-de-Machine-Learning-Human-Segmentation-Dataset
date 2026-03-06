from ultralytics import YOLO

# carregar modelo treinado
model = YOLO("runs/segment/train/weights/best.pt")

# rodar predição
results = model.predict(
    source="data/images/val",
    save=True,
    conf=0.25
)

print("Predições salvas em runs/segment/predict")
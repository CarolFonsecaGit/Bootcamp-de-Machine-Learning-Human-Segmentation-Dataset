from ultralytics import YOLO

# carregar modelo treinado
model = YOLO("runs/segment/train/weights/best.pt")

# rodar validação
metrics = model.val()

# mostrar métricas principais
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)

print("Mask mAP50:", metrics.seg.map50)
print("Mask mAP50-95:", metrics.seg.map)
import os
import cv2
import numpy as np

BASE_PATH = "data"

def mask_to_yolo_polygon(mask_path, label_path, class_id=0):
    mask = cv2.imread(mask_path, 0)
    if mask is None:
        print(f"Erro ao ler {mask_path}")
        return

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape

    with open(label_path, "w") as f:
        for contour in contours:
            if len(contour) < 3:
                continue

            contour = contour.squeeze()

            if len(contour.shape) < 2:
                continue

            normalized = []

            for point in contour:
                x, y = point
                normalized.append(x / w)
                normalized.append(y / h)

            line = str(class_id) + " " + " ".join(map(str, normalized))
            f.write(line + "\n")


def convert_split(split):
    masks_dir = os.path.join(BASE_PATH, "masks", split)
    labels_dir = os.path.join(BASE_PATH, "labels", split)

    os.makedirs(labels_dir, exist_ok=True)

    for file in os.listdir(masks_dir):
        if file.endswith(".png"):
            mask_path = os.path.join(masks_dir, file)
            label_path = os.path.join(labels_dir, file.replace(".png", ".txt"))

            mask_to_yolo_polygon(mask_path, label_path)

    print(f"✅ Conversão concluída para {split}")


convert_split("train")
convert_split("val")

print("🎯 Todas as máscaras foram convertidas.")
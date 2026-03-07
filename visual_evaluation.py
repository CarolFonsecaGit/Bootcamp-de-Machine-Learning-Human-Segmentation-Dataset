import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

MODEL_PATH = "runs/segment/train/weights/best.pt"

IMAGE_FOLDER = "data/images/val"
OUTPUT_FOLDER = "avaliacao_refinamento"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

model = YOLO(MODEL_PATH)


# FUNÇÃO DE REFINAMENTO

def refinar_mascara(mask):

    mask = mask.astype(np.uint8)

    kernel = np.ones((5,5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask_refinada = np.zeros_like(mask)

    cv2.drawContours(mask_refinada, contornos, -1, 255, thickness=-1)

    return mask_refinada


# VISUALIZAÇÃO

def salvar_comparacao(img, mask_yolo, mask_refinada, nome):

    fig, ax = plt.subplots(1,3, figsize=(15,5))

    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Original")
    ax[0].axis("off")

    ax[1].imshow(mask_yolo, cmap="gray")
    ax[1].set_title("YOLO")
    ax[1].axis("off")

    ax[2].imshow(mask_refinada, cmap="gray")
    ax[2].set_title("Refinada")
    ax[2].axis("off")

    save_path = os.path.join(OUTPUT_FOLDER, nome)

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()



for img_name in os.listdir(IMAGE_FOLDER):

    img_path = os.path.join(IMAGE_FOLDER, img_name)

    if not os.path.isfile(img_path):
        continue

    if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img = cv2.imread(img_path)

    if img is None:
        continue

    results = model(img)

    if results[0].masks is None:
        continue

    masks = results[0].masks.data.cpu().numpy()

    for i, mask in enumerate(masks):

        mask = (mask * 255).astype(np.uint8)

        # garantir mesmo tamanho da imagem
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

        mask_refined = refinar_mascara(mask)

        nome_saida = f"{img_name.split('.')[0]}_{i}.png"

        salvar_comparacao(img, mask, mask_refined, nome_saida)

print("Avaliação visual finalizada!")
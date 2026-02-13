import os
import numpy as np
from PIL import Image

print("--- Análise Real de Distribuição das Classes (Segmentação) ---")

# Caminho base
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MASKS_PATH = os.path.join(BASE_PATH, "segmentation_full_body_tik_tok_2615_img", "masks")

pixel_counts = {}

# Percorrer todas as máscaras
for mask_file in os.listdir(MASKS_PATH):
    mask_path = os.path.join(MASKS_PATH, mask_file)

    mask = Image.open(mask_path)
    mask_array = np.array(mask)

    unique, counts = np.unique(mask_array, return_counts=True)

    for value, count in zip(unique, counts):
        pixel_counts[value] = pixel_counts.get(value, 0) + count

# Calcular total de pixels
total_pixels = sum(pixel_counts.values())

print("\nDistribuição de pixels por classe:")
for classe, count in pixel_counts.items():
    percentual = (count / total_pixels) * 100
    print(f"Classe {classe}: {percentual:.2f}%")

print("\nTotal de máscaras analisadas:", len(os.listdir(MASKS_PATH)))

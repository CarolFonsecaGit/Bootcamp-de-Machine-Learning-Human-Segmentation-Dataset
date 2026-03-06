from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

model = YOLO("runs/segment/train/weights/best.pt")

img_path = "data/images/val"
mask_path = "data/masks/val"

img_name = os.listdir(img_path)[0]

image = cv2.imread(os.path.join(img_path, img_name))
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

gt_mask = cv2.imread(os.path.join(mask_path, img_name.replace(".png",".png")), 0)

results = model(os.path.join(img_path, img_name))

pred_mask = results[0].masks.data[0].cpu().numpy()
pred_mask = (pred_mask > 0.5).astype(np.uint8)

pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]))

overlay = image_rgb.copy()
overlay[pred_mask==1] = [255,0,0]

plt.figure(figsize=(12,4))

plt.subplot(1,4,1)
plt.imshow(image_rgb)
plt.title("Imagem")

plt.subplot(1,4,2)
plt.imshow(gt_mask, cmap="gray")
plt.title("Máscara real")

plt.subplot(1,4,3)
plt.imshow(pred_mask, cmap="gray")
plt.title("Máscara prevista")

plt.subplot(1,4,4)
plt.imshow(overlay)
plt.title("Segmentação")

plt.show()
from ultralytics import YOLO
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

model = YOLO("runs/segment/train/weights/best.pt")

images_dir = "data/images/val"
masks_dir = "data/masks/val"

ious = []
dices = []

for img_name in os.listdir(images_dir):

    img_path = os.path.join(images_dir, img_name)
    gt_path = os.path.join(masks_dir, img_name)

    if not os.path.exists(gt_path):
        continue

    image = cv2.imread(img_path)

    results = model(img_path)

    # pular se não houver detecção
    if results[0].masks is None or len(results[0].masks.data) == 0:
        continue

    pred_mask = results[0].masks.data[0].cpu().numpy()
    pred_mask = (pred_mask > 0.5).astype(np.uint8)

    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    gt_mask = (gt_mask > 0).astype(np.uint8)

    gt_mask = np.squeeze(gt_mask)

    pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]))

    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()

    if union == 0:
        continue

    iou = intersection / union
    dice = (2 * intersection) / (pred_mask.sum() + gt_mask.sum())

    ious.append(iou)
    dices.append(dice)


print("IoU médio:", np.mean(ious))
print("Dice médio:", np.mean(dices))


# GRÁFICOS

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.hist(ious, bins=20)
plt.title("Distribuição IoU")
plt.xlabel("IoU")
plt.ylabel("Quantidade")

plt.subplot(1,2,2)
plt.hist(dices, bins=20)
plt.title("Distribuição Dice")
plt.xlabel("Dice")
plt.ylabel("Quantidade")

plt.tight_layout()
plt.show()
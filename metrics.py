from ultralytics import YOLO
import cv2
import numpy as np
import os

model = YOLO("runs/segment/train/weights/best.pt")

val_images = "data/images/val"
gt_masks = "data/masks/val"

ious = []
dices = []

for img_name in os.listdir(val_images):

    img_path = os.path.join(val_images, img_name)
    gt_path = os.path.join(gt_masks, img_name.replace(".jpg", ".png"))

    if not os.path.exists(gt_path):
        continue

    results = model(img_path)

    if results[0].masks is None:
        continue

    if results[0].masks is None or len(results[0].masks.data) == 0:
        continue

    pred_mask = results[0].masks.data[0].cpu().numpy()
    pred_mask = (pred_mask > 0.5).astype(np.uint8)

    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    gt_mask = (gt_mask > 0).astype(np.uint8)

    # garantir que não tenha canal extra
    gt_mask = np.squeeze(gt_mask)

    # ajustar tamanho da máscara prevista
    pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]))

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
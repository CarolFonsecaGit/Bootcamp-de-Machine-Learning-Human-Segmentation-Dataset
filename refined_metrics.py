from ultralytics import YOLO
import cv2
import numpy as np
import os

model = YOLO("runs/segment/train/weights/best.pt")

images_dir = "data/images/val"
masks_dir = "data/masks/val"

ious = []
dices = []

def refine_mask(mask):

    mask = mask.astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return mask

    largest = max(contours, key=cv2.contourArea)

    refined = np.zeros_like(mask)

    cv2.drawContours(refined, [largest], -1, 255, thickness=-1)

    refined = refined > 0

    return refined


for img_name in os.listdir(images_dir):

    img_path = os.path.join(images_dir, img_name)
    gt_path = os.path.join(masks_dir, img_name)

    if not os.path.exists(gt_path):
        continue

    results = model(img_path)

    if results[0].masks is None or len(results[0].masks.data) == 0:
        continue

    pred_mask = results[0].masks.data[0].cpu().numpy()

    pred_mask = (pred_mask > 0.5).astype(np.uint8)

    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    gt_mask = (gt_mask > 0)

    gt_mask = np.squeeze(gt_mask)

    pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]))

    pred_mask = pred_mask.astype(bool)

    # 🔵 AQUI entra o refinamento
    refined_mask = refine_mask(pred_mask)

    intersection = np.logical_and(refined_mask, gt_mask).sum()
    union = np.logical_or(refined_mask, gt_mask).sum()

    if union == 0:
        continue

    iou = intersection / union
    dice = (2 * intersection) / (refined_mask.sum() + gt_mask.sum())

    ious.append(iou)
    dices.append(dice)


print("IoU médio refinado:", np.mean(ious))
print("Dice médio refinado:", np.mean(dices))
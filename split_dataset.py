import os
import random
import shutil

# Caminhos
BASE_PATH = "data"
IMAGES_PATH = os.path.join(BASE_PATH, "images")
MASKS_PATH = os.path.join(BASE_PATH, "masks")

# Criar pastas
for split in ["train", "val"]:
    os.makedirs(os.path.join(BASE_PATH, "images", split), exist_ok=True)
    os.makedirs(os.path.join(BASE_PATH, "masks", split), exist_ok=True)

# Lista imagens
images = os.listdir(IMAGES_PATH)
images = [img for img in images if img.endswith(".png")]

random.shuffle(images)

split_index = int(0.8 * len(images))
train_images = images[:split_index]
val_images = images[split_index:]

def move_files(image_list, split):
    for img in image_list:
        mask_name = img.replace(".png", ".png")

        shutil.move(
            os.path.join(IMAGES_PATH, img),
            os.path.join(BASE_PATH, "images", split, img)
        )

        shutil.move(
            os.path.join(MASKS_PATH, mask_name),
            os.path.join(BASE_PATH, "masks", split, mask_name)
        )

move_files(train_images, "train")
move_files(val_images, "val")

print("Divisão concluída ✅")
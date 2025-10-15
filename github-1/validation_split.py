import os
import random
import shutil
import yaml
from collections import defaultdict

# ---------------- CONFIG ----------------
root = os.getcwd()
src_img = os.path.join(root, "cleaned_dataset", "images")
src_lbl = os.path.join(root, "cleaned_dataset", "labels")
out_base = os.path.join(root, "datasets", "weld_defects")
train_ratio = 0.8

os.makedirs(os.path.join(out_base, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(out_base, "images", "val"), exist_ok=True)
os.makedirs(os.path.join(out_base, "labels", "train"), exist_ok=True)
os.makedirs(os.path.join(out_base, "labels", "val"), exist_ok=True)

classes = ["air hole", "bite edge", "broken arc", "crack", "overlap", "slag inclusion", "unfused"]

# ---------------- STRATIFIED SPLIT ----------------
# Group images by main class (first label in file)
class_to_imgs = defaultdict(list)

for lbl_file in os.listdir(src_lbl):
    if not lbl_file.endswith(".txt"):
        continue
    lbl_path = os.path.join(src_lbl, lbl_file)
    with open(lbl_path, "r") as f:
        lines = f.readlines()
    if not lines:
        continue
    first_cls = int(lines[0].split()[0])  # class id
    class_to_imgs[first_cls].append(lbl_file.replace(".txt", ".jpg"))

train_imgs, val_imgs = [], []

for cls_id, img_list in class_to_imgs.items():
    random.shuffle(img_list)
    split = int(len(img_list) * train_ratio)
    train_imgs += img_list[:split]
    val_imgs += img_list[split:]

print(f"Stratified split complete.")
print(f"Train: {len(train_imgs)} | Val: {len(val_imgs)}")

def move_files(img_list, split):
    for img_name in img_list:
        lbl_name = os.path.splitext(img_name)[0] + ".txt"
        img_src = os.path.join(src_img, img_name)
        lbl_src = os.path.join(src_lbl, lbl_name)
        if not os.path.exists(img_src) or not os.path.exists(lbl_src):
            continue
        shutil.copy2(img_src, os.path.join(out_base, "images", split, img_name))
        shutil.copy2(lbl_src, os.path.join(out_base, "labels", split, lbl_name))

move_files(train_imgs, "train")
move_files(val_imgs, "val")

# ---------------- YAML CREATION ----------------
yaml_path = os.path.join(root, "weld_defects.yaml")
data = {
    "train": os.path.join("datasets", "weld_defects", "images", "train"),
    "val": os.path.join("datasets", "weld_defects", "images", "val"),
    "nc": len(classes),
    "names": classes
}
with open(yaml_path, "w") as f:
    yaml.dump(data, f, sort_keys=False)

print(f"YAML created at {yaml_path}")

import os
import json
import shutil

# ---------------- CONFIG ----------------
root = os.getcwd()  # use current directory automatically
src_img_dir = os.path.join(root, "images")
src_json_dir = os.path.join(root, "json")
out_base = os.path.join(root, "cleaned_dataset")

# output structure
os.makedirs(os.path.join(out_base, "images"), exist_ok=True)
os.makedirs(os.path.join(out_base, "labels"), exist_ok=True)

# unified class list
classes = [
    "air hole",
    "bite edge",
    "broken arc",
    "crack",
    "overlap",
    "slag inclusion",
    "unfused"
]

# normalization helper
def normalize_name(name: str):
    """Normalize class names like 'slag-inclusion' -> 'slag inclusion'"""
    return name.strip().lower().replace("-", " ").replace("_", " ")

# ---------------- MAIN ----------------
json_files = [f for f in os.listdir(src_json_dir) if f.endswith(".json")]
kept, skipped = 0, 0

for js_name in json_files:
    js_path = os.path.join(src_json_dir, js_name)
    try:
        with open(js_path, "r") as f:
            data = json.load(f)

        # match image name by file stem (ignore JSON 'path' field completely)
        base = os.path.splitext(js_name)[0]
        img_name = base + ".jpg"
        img_src = os.path.join(src_img_dir, img_name)

        if not os.path.exists(img_src):
            # fallback to PNG
            img_name = base + ".png"
            img_src = os.path.join(src_img_dir, img_name)
            if not os.path.exists(img_src):
                print(f"Missing image for {js_name}")
                skipped += 1
                continue

        # skip augmented variants like (flip), (crop)
        if "(" in img_name or ")" in img_name:
            continue

        # dimensions
        width = data["size"]["width"]
        height = data["size"]["height"]

        objects = data.get("outputs", {}).get("object", [])
        if not objects:
            print(f"No objects found in {js_name}")
            skipped += 1
            continue

        label_lines = []
        for obj in objects:
            cls_raw = obj["name"]
            cls = normalize_name(cls_raw)

            if cls not in classes:
                print(f"New class '{cls}' found, adding dynamically.")
                classes.append(cls)
            cls_id = classes.index(cls)

            bb = obj["bndbox"]
            xmin, ymin, xmax, ymax = bb["xmin"], bb["ymin"], bb["xmax"], bb["ymax"]

            # normalize for YOLO
            x_c = ((xmin + xmax) / 2) / width
            y_c = ((ymin + ymax) / 2) / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height
            label_lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

        # copy image
        img_dst = os.path.join(out_base, "images", img_name)
        shutil.copy2(img_src, img_dst)

        # write YOLO label
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        label_dst = os.path.join(out_base, "labels", txt_name)
        with open(label_dst, "w") as f:
            f.writelines(label_lines)

        kept += 1

    except Exception as e:
        print(f"❌ Error in {js_name}: {e}")
        skipped += 1

print(f"\nCleaning complete!")
print(f"Kept {kept} valid image–label pairs | Skipped {skipped}.")
print(f"Classes used: {classes}")

import os
import shutil
import pandas as pd

CSV_PATH = "labels.csv"
IMAGE_DIR = "images"
OUT_DIR = "edgeimpulse_dataset"

USE_ACTIONS = ["LEFT", "STRAIGHT", "RIGHT"]

df = pd.read_csv(CSV_PATH)

# 建立 image filename -> action 的查找表
label_map = dict(zip(df["image"], df["action"]))

os.makedirs(OUT_DIR, exist_ok=True)
for action in USE_ACTIONS:
    os.makedirs(os.path.join(OUT_DIR, action), exist_ok=True)

valid_exts = (".jpg", ".jpeg", ".png", ".bmp")

copied = 0
missing_label = 0
skipped_action = 0

for image_name in os.listdir(IMAGE_DIR):
    if not image_name.lower().endswith(valid_exts):
        continue

    action = label_map.get(image_name)

    if action is None:
        print("No label in CSV:", image_name)
        missing_label += 1
        continue

    if action not in USE_ACTIONS:
        skipped_action += 1
        continue

    src = os.path.join(IMAGE_DIR, image_name)
    dst = os.path.join(OUT_DIR, action, image_name)

    shutil.copy2(src, dst)
    copied += 1

print("Done.")
print("Copied:", copied)
print("Images without CSV label:", missing_label)
print("Skipped actions:", skipped_action)

print("\nCSV label counts:")
print(df["action"].value_counts())

print("\nOutput label counts:")
for action in USE_ACTIONS:
    folder = os.path.join(OUT_DIR, action)
    print(action, len(os.listdir(folder)))
"""
prepare_plantdoc_yolo.py
Converts PlantDoc dataset annotations (CSV/XML) into normalized YOLOv8 format:
<class_id> <x_center> <y_center> <width> <height>
Stages images and labels into cv/datasets/processed/yolo_plantdoc/
and exports cv/configs/yolo_plantdoc.yaml.
"""

import os
import shutil
from pathlib import Path
import pandas as pd
import yaml

print("=" * 85)
print("📦 CONVERTING & STAGING PLANTDOC FIELD DATASET FOR YOLOV8")
print("=" * 85)

plantdoc_raw_dir = Path("cv/datasets/raw/plantdoc_od_repo")
yolo_out_dir = Path("cv/datasets/processed/yolo_plantdoc")

for split in ["train", "val"]:
    (yolo_out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (yolo_out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

# Find label CSV files
train_csv = plantdoc_raw_dir / "train_labels.csv"
test_csv = plantdoc_raw_dir / "test_labels.csv"

if not train_csv.exists() or not test_csv.exists():
    print(f"[*] Waiting or searching for annotation CSVs in {plantdoc_raw_dir}...")
    csvs = list(plantdoc_raw_dir.glob("*.csv"))
    print(f"    Found CSVs: {[c.name for c in csvs]}")

def process_split(csv_path, split_name, img_dir):
    if not csv_path.exists():
        print(f"[-] Missing {csv_path}")
        return {}
    
    df = pd.read_csv(csv_path)
    # Expected cols: filename, width, height, class, xmin, ymin, xmax, ymax
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Get unique classes
    unique_classes = sorted(df["class"].dropna().unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    
    grouped = df.groupby("filename")
    total_imgs = 0
    total_boxes = 0
    
    for filename, group in grouped:
        src_img = img_dir / filename
        if not src_img.exists():
            # Try subfolder search
            matches = list(img_dir.rglob(filename))
            if matches:
                src_img = matches[0]
            else:
                continue
        
        dst_img = yolo_out_dir / "images" / split_name / filename
        shutil.copy2(src_img, dst_img)
        
        txt_path = yolo_out_dir / "labels" / split_name / f"{Path(filename).stem}.txt"
        lines = []
        
        for _, row in group.iterrows():
            c_name = row["class"]
            if c_name not in class_to_idx:
                continue
            cid = class_to_idx[c_name]
            
            w = float(row["width"])
            h = float(row["height"])
            if w <= 0 or h <= 0:
                continue
            
            xmin = max(0.0, float(row["xmin"]))
            ymin = max(0.0, float(row["ymin"]))
            xmax = min(w, float(row["xmax"]))
            ymax = min(h, float(row["ymax"]))
            
            if xmax <= xmin or ymax <= ymin:
                continue
            
            xc = ((xmin + xmax) / 2.0) / w
            yc = ((ymin + ymax) / 2.0) / h
            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h
            
            lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            total_boxes += 1
            
        with open(txt_path, "w") as f:
            f.write("\n".join(lines))
        total_imgs += 1
        
    print(f"[*] Processed {split_name}: {total_imgs} images, {total_boxes} boxes across {len(unique_classes)} classes.")
    return class_to_idx

# Process train and val
train_img_dir = plantdoc_raw_dir / "TRAIN" if (plantdoc_raw_dir / "TRAIN").exists() else plantdoc_raw_dir / "train"
test_img_dir = plantdoc_raw_dir / "TEST" if (plantdoc_raw_dir / "TEST").exists() else plantdoc_raw_dir / "test"

c_map = process_split(train_csv, "train", train_img_dir)
process_split(test_csv, "val", test_img_dir)

if c_map:
    yaml_dict = {
        "path": str(yolo_out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {v: k for k, v in c_map.items()}
    }
    
    yaml_path = Path("cv/configs/yolo_plantdoc.yaml")
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_dict, f, sort_keys=False)
        
    print(f"[*] Exported YOLO configuration to {yaml_path.resolve()}")
    print("=" * 85)

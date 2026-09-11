"""
prepare_manifest_38classes.py
Scans the 38 classes of PlantVillage, executes 4-gate validation,
and generates a stratified 70/15/15 dataset partition manifest.
"""

import json
from pathlib import Path
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

print("=" * 85)
print("📦 AUDITING & PARTITIONING 38-CLASS PLANTVILLAGE DATASET")
print("=" * 85)

taxonomy_path = Path("cv/configs/taxonomy_38classes.json")
with open(taxonomy_path) as f:
    taxonomy = json.load(f)

folder_to_meta = {item["raw_folder"]: item for item in taxonomy}
raw_color_dir = Path("cv/datasets/raw/plantvillage/raw/color")

records = []
total_corrupt = 0

print(f"[*] Scanning images across {len(taxonomy)} canonical classes...")

for folder_name, meta in folder_to_meta.items():
    folder_path = raw_color_dir / folder_name
    if not folder_path.exists():
        continue
    
    img_files = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.JPG")) + list(folder_path.glob("*.png"))
    valid_in_class = 0

    for img_p in img_files:
        try:
            # 4-gate image validation
            sz = img_p.stat().st_size
            if sz < 1024:  # Gate 1: Non-zero size (>1KB)
                total_corrupt += 1
                continue
            
            with Image.open(img_p) as im:
                w, h = im.size
                if w < 64 or h < 64:  # Gate 2: Dimension check
                    total_corrupt += 1
                    continue
                if im.mode not in ("RGB", "L"):  # Gate 3: Channels
                    total_corrupt += 1
                    continue
            
            records.append({
                "file_path": str(img_p.resolve()),
                "filename": img_p.name,
                "crop": meta["crop"],
                "condition_type": meta["condition_type"],
                "disease_name": meta["disease_name"],
                "scientific_name": meta["scientific_name"],
                "canonical_id": meta["canonical_id"],
                "class_index": meta["class_index"],
                "file_size_bytes": sz,
                "width": w,
                "height": h
            })
            valid_in_class += 1
        except Exception:
            total_corrupt += 1

df = pd.DataFrame(records)
print(f"[*] Valid Images Verified     : {len(df):,} images")
print(f"[*] Corrupt/Invalid Discarded : {total_corrupt} files")
print(f"[*] Classes Present on Disk   : {df['canonical_id'].nunique()} / 38 classes")

# ── Stratified 70 / 15 / 15 Partitioning ──────────────────────────────────────
print("-" * 85)
print("[*] Executing Stratified 70/15/15 Partitioning across all classes...")

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["class_index"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["class_index"]
)

train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

partitioned_df = pd.concat([train_df, val_df, test_df]).sort_index()

out_manifest = Path("cv/datasets/processed/manifest_38classes_partitioned.csv")
out_manifest.parent.mkdir(parents=True, exist_ok=True)
partitioned_df.to_csv(out_manifest, index=False)

print(f"  • Train Split (70%)         : {len(train_df):,} images")
print(f"  • Val Split   (15%)         : {len(val_df):,} images")
print(f"  • Test Split  (15%)         : {len(test_df):,} images")
print(f"[*] Saved Manifest to         : {out_manifest.resolve()}")
print("=" * 85)

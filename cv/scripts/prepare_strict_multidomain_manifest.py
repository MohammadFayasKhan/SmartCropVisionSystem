"""
prepare_strict_multidomain_manifest.py
--------------------------------------
Constructs a strict, leakage-free multi-domain dataset manifest:
1. PlantDoc Field Images:
   - plantdoc_repo/train: Stratified split into 85% train and 15% val.
   - plantdoc_repo/test: Strictly held-out test set (236 images).
2. PlantVillage Lab Images:
   - plantvillage train split: Stratified sample (up to 120 per class).
   - plantvillage val split: Stratified sample (up to 30 per class).
   - plantvillage test split: 8,146 strictly held-out lab test images.
3. Output saved to cv/datasets/processed/manifest_38classes_multidomain.csv.
"""

import json
from pathlib import Path
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

print("=" * 80)
print("🌿 PREPARING LEAKAGE-FREE MULTI-DOMAIN 38-CLASS MANIFEST")
print("=" * 80)

tax_path = Path("cv/configs/taxonomy_38classes.json")
with open(tax_path) as f:
    taxonomy = json.load(f)

raw_to_meta = {item["raw_folder"]: item for item in taxonomy}

PLANTDOC_TO_TAXONOMY = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot"
}

# 1. Ingest PlantDoc
plantdoc_base = Path("cv/datasets/raw/plantdoc_repo")
plantdoc_train_records = []
plantdoc_test_records = []

for split_folder in ["train", "test"]:
    split_dir = plantdoc_base / split_folder
    if not split_dir.exists():
        continue
    for class_dir in split_dir.iterdir():
        if not class_dir.is_dir() or class_dir.name not in PLANTDOC_TO_TAXONOMY:
            continue
        pv_folder = PLANTDOC_TO_TAXONOMY[class_dir.name]
        meta = raw_to_meta[pv_folder]

        for img_p in list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.JPG")) + list(class_dir.glob("*.png")):
            try:
                if img_p.stat().st_size < 512:
                    continue
                with Image.open(img_p) as im:
                    w, h = im.size
                    if w < 48 or h < 48:
                        continue
                rec = {
                    "file_path": str(img_p.resolve()),
                    "filename": img_p.name,
                    "crop": meta["crop"],
                    "condition_type": meta["condition_type"],
                    "disease_name": meta["disease_name"],
                    "scientific_name": meta["scientific_name"],
                    "canonical_id": meta["canonical_id"],
                    "class_index": meta["class_index"],
                    "source": "plantdoc_field",
                    "domain": "field"
                }
                if split_folder == "train":
                    plantdoc_train_records.append(rec)
                else:
                    rec["split"] = "test"
                    plantdoc_test_records.append(rec)
            except Exception:
                continue

df_pd_train = pd.DataFrame(plantdoc_train_records)
df_pd_test = pd.DataFrame(plantdoc_test_records)

print(f"[*] PlantDoc Valid Field Images: {len(df_pd_train)} train, {len(df_pd_test)} held-out test")

# Stratified split of plantdoc train into train (85%) and val (15%)
train_idx, val_idx = train_test_split(
    df_pd_train.index,
    test_size=0.15,
    random_state=42,
    stratify=df_pd_train["class_index"]
)
df_pd_train_split = df_pd_train.loc[train_idx].copy()
df_pd_train_split["split"] = "train"

df_pd_val_split = df_pd_train.loc[val_idx].copy()
df_pd_val_split["split"] = "val"

print(f"  • Field Train Split (85%)   : {len(df_pd_train_split):,} images")
print(f"  • Field Val Split (15%)     : {len(df_pd_val_split):,} images")
print(f"  • Field Test Split (Held-out): {len(df_pd_test):,} images")

# 2. Ingest PlantVillage
pv_manifest = Path("cv/datasets/processed/manifest_38classes_partitioned.csv")
pv_df = pd.read_csv(pv_manifest)
pv_df["source"] = "plantvillage_lab"
pv_df["domain"] = "lab"

# Sample up to 120 per class for training, 30 per class for validation
pv_train_sub = []
pv_val_sub = []
pv_test_sub = pv_df[pv_df["split"] == "test"].copy()

for c_idx, group in pv_df.groupby("class_index"):
    tr_grp = group[group["split"] == "train"]
    va_grp = group[group["split"] == "val"]
    pv_train_sub.append(tr_grp.sample(n=min(len(tr_grp), 120), random_state=42))
    pv_val_sub.append(va_grp.sample(n=min(len(va_grp), 30), random_state=42))

pv_train_df = pd.concat(pv_train_sub, ignore_index=True)
pv_val_df = pd.concat(pv_val_sub, ignore_index=True)

print(f"[*] PlantVillage Sampled Lab Images:")
print(f"  • Lab Train Split           : {len(pv_train_df):,} images (across 38 classes)")
print(f"  • Lab Val Split             : {len(pv_val_df):,} images (across 38 classes)")
print(f"  • Lab Test Split (Held-out) : {len(pv_test_sub):,} images (across 38 classes)")

# 3. Add Real-World Field User Leaf Specimens if present
user_healthy_paths = [
    Path("cv/datasets/processed/segmentation/images/val/0001.jpg"),
    Path("cv/datasets/processed/segmentation/images/val/0002.jpg")
]
user_records = []
for p in user_healthy_paths:
    if p.exists():
        meta = raw_to_meta["Tomato___healthy"]
        user_records.append({
            "file_path": str(p.resolve()),
            "filename": p.name,
            "crop": meta["crop"],
            "condition_type": meta["condition_type"],
            "disease_name": meta["disease_name"],
            "scientific_name": meta["scientific_name"],
            "canonical_id": meta["canonical_id"],
            "class_index": meta["class_index"],
            "source": "field_user_specimen",
            "domain": "field",
            "split": "train"
        })
df_user = pd.DataFrame(user_records)

# 4. Concatenate Full Multi-Domain Dataset
common_cols = [
    "file_path", "filename", "crop", "condition_type", "disease_name",
    "scientific_name", "canonical_id", "class_index", "source", "domain", "split"
]

all_parts = [
    df_pd_train_split[common_cols],
    df_pd_val_split[common_cols],
    df_pd_test[common_cols],
    pv_train_df[common_cols],
    pv_val_df[common_cols],
    pv_test_sub[common_cols]
]
if len(df_user) > 0:
    all_parts.append(df_user[common_cols])

final_manifest = pd.concat(all_parts, ignore_index=True)

out_manifest = Path("cv/datasets/processed/manifest_38classes_multidomain.csv")
out_manifest.parent.mkdir(parents=True, exist_ok=True)
final_manifest.to_csv(out_manifest, index=False)

print("\n" + "=" * 80)
print(f"✅ EXPORTED MULTI-DOMAIN MANIFEST TO: {out_manifest.resolve()}")
print(f"[*] Total Records             : {len(final_manifest):,}")
print(f"[*] Total Classes Represented : {final_manifest['class_index'].nunique()} / 38")
print("\n[Breakdown by Split & Domain]")
print(final_manifest.groupby(["split", "domain", "source"]).size())
print("=" * 80)

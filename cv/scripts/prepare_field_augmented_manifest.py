"""
prepare_field_augmented_manifest.py
-----------------------------------
Creates a balanced field-augmented training manifest combining:
1. PlantDoc real-world field images (28 classes mapped to canonical 38-class taxonomy).
2. Stratified sample of PlantVillage laboratory images (across all 38 classes).
3. Field healthy foliage samples (including real-world tomato leaves).
"""

import json
from pathlib import Path
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

print("=" * 80)
print("📦 CREATING BALANCED FIELD-AUGMENTED 38-CLASS MANIFEST")
print("=" * 80)

# 1. Load Taxonomy
tax_path = Path("cv/configs/taxonomy_38classes.json")
with open(tax_path) as f:
    taxonomy = json.load(f)

raw_to_meta = {item["raw_folder"]: item for item in taxonomy}
idx_to_meta = {item["class_index"]: item for item in taxonomy}

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

records = []

# 2. Ingest PlantDoc Field Images
plantdoc_base = Path("cv/datasets/raw/plantdoc_repo")
for split_folder, split_type in [("train", "train"), ("test", "val")]:
    split_dir = plantdoc_base / split_folder
    if not split_dir.exists():
        continue
    for class_dir in split_dir.iterdir():
        if not class_dir.is_dir():
            continue
        c_name = class_dir.name
        if c_name not in PLANTDOC_TO_TAXONOMY:
            continue
        pv_folder = PLANTDOC_TO_TAXONOMY[c_name]
        meta = raw_to_meta[pv_folder]

        for img_p in list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.JPG")) + list(class_dir.glob("*.png")):
            try:
                if img_p.stat().st_size < 512:
                    continue
                with Image.open(img_p) as im:
                    w, h = im.size
                    if w < 48 or h < 48:
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
                    "source": "plantdoc_field",
                    "split": split_type
                })
            except Exception:
                continue

plantdoc_count = len(records)
print(f"[*] Ingested {plantdoc_count} real-world PlantDoc field images.")

# Ingest curated healthy leaf samples from local dataset if present
downloads_healthy = [
    Path("cv/datasets/processed/segmentation/images/val/0001.jpg"),
    Path("cv/datasets/processed/segmentation/images/val/0002.jpg")
]
for p in downloads_healthy:
    if p.exists():
        meta = raw_to_meta["Tomato___healthy"]
        records.append({
            "file_path": str(p.resolve()),
            "filename": p.name,
            "crop": meta["crop"],
            "condition_type": meta["condition_type"],
            "disease_name": meta["disease_name"],
            "scientific_name": meta["scientific_name"],
            "canonical_id": meta["canonical_id"],
            "class_index": meta["class_index"],
            "source": "field_healthy_user",
            "split": "train"
        })

# 3. Ingest Balanced PlantVillage Laboratory Dataset
pv_manifest = Path("cv/datasets/processed/manifest_38classes_partitioned.csv")
if pv_manifest.exists():
    pv_df = pd.read_csv(pv_manifest)
    print(f"[*] Loaded PlantVillage base manifest: {len(pv_df)} images.")
    # Sample up to 100 images per class for training, and 25 per class for validation
    pv_sampled = []
    for c_idx, group in pv_df.groupby("class_index"):
        train_grp = group[group["split"] == "train"]
        val_grp = group[group["split"] == "val"]
        s_train = train_grp.sample(n=min(len(train_grp), 80), random_state=42)
        s_val = val_grp.sample(n=min(len(val_grp), 20), random_state=42)
        pv_sampled.append(s_train)
        pv_sampled.append(s_val)
    pv_combined = pd.concat(pv_sampled, ignore_index=True)
    pv_combined["source"] = "plantvillage_lab"
    
    # Standardize columns
    common_cols = ["file_path", "filename", "crop", "condition_type", "disease_name", "scientific_name", "canonical_id", "class_index", "source", "split"]
    df_field = pd.DataFrame(records)
    df_pv = pv_combined[common_cols]
    df_final = pd.concat([df_field, df_pv], ignore_index=True)
else:
    df_final = pd.DataFrame(records)

out_manifest = Path("cv/datasets/processed/manifest_38classes_field_augmented.csv")
out_manifest.parent.mkdir(parents=True, exist_ok=True)
df_final.to_csv(out_manifest, index=False)

print(f"\n[+] Successfully exported field-augmented manifest:")
print(f"    - Destination: {out_manifest.resolve()}")
print(f"    - Total Images: {len(df_final):,}")
print(f"    - Split Breakdown: Train={len(df_final[df_final['split']=='train']):,}, Val={len(df_final[df_final['split']=='val']):,}")
print(f"    - Source Breakdown:\n{df_final['source'].value_counts()}")
print(f"    - Total Classes Represented: {df_final['class_index'].nunique()} / 38")

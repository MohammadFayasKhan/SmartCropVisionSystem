# SmartCropVision Dataset Provenance & Data Architecture

## 1. Multi Domain Dataset Integration

SmartCropVision addresses the domain collapse problem common to agricultural machine learning. Laboratory datasets feature clean studio lighting and uniform gray backgrounds, causing models trained on them to fail when deployed in actual agricultural fields. To build robust representations, the project integrates three complementary botanical datasets:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Botanical Data Sources                          │
├─────────────────────────┬────────────────────────┬─────────────────────┤
│ PlantVillage            │ PlantDoc               │ PlantWild           │
│ (Controlled Studio)     │ (Field Canopy Clutter) │ (In The Wild Field) │
│ 54,305 photographs     │ 2,592 outdoor images   │ 18,542 wild images  │
│ 38 canonical classes    │ 29 spatial categories  │ Natural illumination│
└─────────────────────────┴────────────────────────┴─────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  Canonical 38 Class Taxonomy Mapping                   │
│                                                                        │
│  dataset label → crop species → condition type → specific disease      │
└────────────────────────────────────────────────────────────────────────┘
```

## 2. Dataset Characterization & Provenance

### 2.1 PlantVillage Dataset
* **Provenance**: Penn State University and EPFL open access collection.
* **Volume**: 54,305 single leaf images across 14 crop species and 38 pathology classes.
* **Annotation Type**: Whole image categorical labels.
* **Role**: Primary morphological foundation for botanical disease taxonomy and baseline feature learning.
* **Limitation**: Controlled laboratory conditions with detached leaves placed on uniform backgrounds. Susceptible to background over fitting if used in isolation.

### 2.2 PlantDoc Dataset
* **Provenance**: Indian Institute of Technology Kharagpur agricultural research benchmark.
* **Volume**: 2,592 natural field photographs with 8,889 object annotations across 29 classes.
* **Annotation Type**: Pascal VOC bounding boxes converted to normalized YOLO format.
* **Role**: Ground truth supervision for Tier 2 YOLO specimen boundary detection in natural foliage canopies.
* **Truthful Annotation Reality**: Object annotations in PlantDoc enclose whole diseased leaves and leaf clusters with a median area ratio of 0.087 to 0.193. They do not annotate individual microscopic fungal spots.

### 2.3 PlantWild Dataset
* **Provenance**: Aggregated agricultural extension field photographs captured under uncontrolled solar lighting, motion blur, and varied growth stages.
* **Volume**: 18,542 real field images.
* **Annotation Type**: Multi domain categorical annotations.
* **Role**: Field domain generalization and cross domain robustness benchmarking.

### 2.4 Foliar Damage Segmentation Dataset
* **Volume**: 520 curated leaf images paired with 3 class pixel masks.
* **Annotation Type**: Discrete semantic masks (Background: 0, Healthy Leaf: 1, Necrotic Lesions: 2).
* **Role**: Supervised training for the Mobile UNet sub pixel segmentation engine.

### 2.5 Agro Climatic Crop Recommendation Dataset
* **Volume**: 2,200 agricultural field records across 22 crop species.
* **Features**: Soil nitrogen, phosphorus, potassium, temperature, relative humidity, soil pH, and rainfall.
* **File Location**: `data/Crop_Recommendation.csv`
* **Role**: Supervised training of the agro climatic crop selection classifier.

## 3. Data Splitting & Leakage Prevention

A critical issue in published agricultural AI papers is patient leaf data leakage, where photographs of the exact same leaf taken from different angles are placed into both training and test partitions. This inflates reported accuracy numbers artificially while the model fails completely on unseen plants.

SmartCropVision prevents data leakage through:
* **Perceptual Hash Deduplication**: Images are converted to grayscale and downsampled to 9x8 pixels to compute 64 bit difference gradients. Near duplicate image pairs with a Hamming distance of 4 or fewer are strictly isolated to the same split.
* **Stratified 70/15/15 Partitioning**: Datasets are partitioned into 70% training, 15% validation, and 15% held out test sets while strictly preserving class frequency proportions.
* **Zero Patient Leakage Verification**: Automated test suites verify that test partition samples share zero hash proximity with the training set.

## 4. Repository Data Policy

In accordance with production repository best practices:
* Multi gigabyte raw image directories (such as the 80,000 uncompressed files in raw downloads) are decoupled from the deployment tree.
* Reproducibility is guaranteed through `DATASET_MANIFEST.json`, `DATASET_SOURCES.json`, `cv/configs/taxonomy_38classes.json`, and pre recorded evaluation scorecards.
* Production inference executes immediately upon cloning using the server side checkpoints in `cv/models/` without requiring developers to download 50 GB of training data.

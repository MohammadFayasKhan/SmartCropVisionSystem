import json
from pathlib import Path

raw_38 = [
    ("Apple___Apple_scab", "apple", "fungal", "Apple Scab", "Venturia inaequalis", "apple_scab"),
    ("Apple___Black_rot", "apple", "fungal", "Black Rot", "Botryosphaeria obtusa", "black_rot"),
    ("Apple___Cedar_apple_rust", "apple", "fungal", "Cedar Apple Rust", "Gymnosporangium juniperi-virginianae", "apple_rust"),
    ("Apple___healthy", "apple", "healthy", "Healthy Apple Leaf", "N/A", None),
    ("Blueberry___healthy", "blueberry", "healthy", "Healthy Blueberry Leaf", "N/A", None),
    ("Cherry_(including_sour)___Powdery_mildew", "cherry", "fungal", "Powdery Mildew", "Podosphaera clandestina", "powdery_mildew"),
    ("Cherry_(including_sour)___healthy", "cherry", "healthy", "Healthy Cherry Leaf", "N/A", None),
    ("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "corn", "fungal", "Gray Leaf Spot", "Cercospora zeae-maydis", "gray_leaf_spot"),
    ("Corn_(maize)___Common_rust_", "corn", "fungal", "Common Rust", "Puccinia sorghi", "common_rust"),
    ("Corn_(maize)___Northern_Leaf_Blight", "corn", "fungal", "Northern Leaf Blight", "Exserohilum turcicum", "northern_leaf_blight"),
    ("Corn_(maize)___healthy", "corn", "healthy", "Healthy Corn Leaf", "N/A", None),
    ("Grape___Black_rot", "grape", "fungal", "Black Rot", "Guignardia bidwellii", "black_rot"),
    ("Grape___Esca_(Black_Measles)", "grape", "fungal", "Esca (Black Measles)", "Phaeomoniella chlamydospora", "esca"),
    ("Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "grape", "fungal", "Leaf Blight (Isariopsis)", "Pseudocercospora cladosporioides", "leaf_blight"),
    ("Grape___healthy", "grape", "healthy", "Healthy Grape Leaf", "N/A", None),
    ("Orange___Haunglongbing_(Citrus_greening)", "orange", "bacterial", "Citrus Greening (HLB)", "Candidatus Liberibacter", "citrus_greening"),
    ("Peach___Bacterial_spot", "peach", "bacterial", "Bacterial Spot", "Xanthomonas arboricola", "bacterial_spot"),
    ("Peach___healthy", "peach", "healthy", "Healthy Peach Leaf", "N/A", None),
    ("Pepper,_bell___Bacterial_spot", "pepper", "bacterial", "Bacterial Spot", "Xanthomonas campestris", "bacterial_spot"),
    ("Pepper,_bell___healthy", "pepper", "healthy", "Healthy Pepper Leaf", "N/A", None),
    ("Potato___Early_blight", "potato", "fungal", "Early Blight", "Alternaria solani", "early_blight"),
    ("Potato___Late_blight", "potato", "fungal", "Late Blight", "Phytophthora infestans", "late_blight"),
    ("Potato___healthy", "potato", "healthy", "Healthy Potato Leaf", "N/A", None),
    ("Raspberry___healthy", "raspberry", "healthy", "Healthy Raspberry Leaf", "N/A", None),
    ("Soybean___healthy", "soybean", "healthy", "Healthy Soybean Leaf", "N/A", None),
    ("Squash___Powdery_mildew", "squash", "fungal", "Powdery Mildew", "Podosphaera xanthii", "powdery_mildew"),
    ("Strawberry___Leaf_scorch", "strawberry", "fungal", "Leaf Scorch", "Diplocarpon earlianum", "leaf_scorch"),
    ("Strawberry___healthy", "strawberry", "healthy", "Healthy Strawberry Leaf", "N/A", None),
    ("Tomato___Bacterial_spot", "tomato", "bacterial", "Bacterial Spot", "Xanthomonas perforans", "bacterial_spot"),
    ("Tomato___Early_blight", "tomato", "fungal", "Early Blight", "Alternaria solani", "early_blight"),
    ("Tomato___Late_blight", "tomato", "fungal", "Late Blight", "Phytophthora infestans", "late_blight"),
    ("Tomato___Leaf_Mold", "tomato", "fungal", "Leaf Mold", "Passalora fulva", "leaf_mold"),
    ("Tomato___Septoria_leaf_spot", "tomato", "fungal", "Septoria Leaf Spot", "Septoria lycopersici", "septoria_spot"),
    ("Tomato___Spider_mites Two-spotted_spider_mite", "tomato", "pest", "Two-Spotted Spider Mite", "Tetranychus urticae", "spider_mite"),
    ("Tomato___Target_Spot", "tomato", "fungal", "Target Spot", "Corynespora cassiicola", "target_spot"),
    ("Tomato___Tomato_Yellow_Leaf_Curl_Virus", "tomato", "viral", "Yellow Leaf Curl Virus", "TYLCV", "yellow_leaf_curl"),
    ("Tomato___Tomato_mosaic_virus", "tomato", "viral", "Mosaic Virus", "ToMV", "mosaic_virus"),
    ("Tomato___healthy", "tomato", "healthy", "Healthy Tomato Leaf", "N/A", None)
]

taxonomy = []
for idx, (raw_name, crop, cond_type, dis_name, sci_name, engine_id) in enumerate(raw_38):
    tail = raw_name.split("___")[-1].lower().replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
    can_id = f"{crop}__{cond_type}__{tail}"
    taxonomy.append({
        "class_index": idx,
        "canonical_id": can_id,
        "raw_folder": raw_name,
        "crop": crop,
        "condition_type": cond_type,
        "disease_name": dis_name,
        "scientific_name": sci_name,
        "linked_engine_id": engine_id
    })

out_path = Path("cv/configs/taxonomy_38classes.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(taxonomy, f, indent=2)

print(f"Exported {len(taxonomy)} canonical classes to {out_path}!")

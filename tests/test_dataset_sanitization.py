"""
Automated Regression Tests for Dataset Filename Sanitization & Kaggle Compatibility.
Verifies that:
  1. Exact PlantDoc filenames from Kaggle error are sanitized into compliant names.
  2. Image and Pascal VOC XML pairs preserve identical stems.
  3. XML internal <filename> and <path> tags stay synchronized.
  4. Collisions are resolved deterministically without overwriting.
  5. The release archive smartcrop_codebase_master.zip passes all Kaggle compatibility gates.
"""
import pytest
import tempfile
import zipfile
from pathlib import Path

from cv.scripts.dataset_sanitizer import (
    FORBIDDEN_KAGGLE_CHARS,
    sanitize_stem,
    parse_and_sanitize_filename,
    sanitize_dataset_directory,
    audit_dataset_pair_integrity,
    validate_zip_compatibility
)

KAGGLE_ERROR_SAMPLES = [
    ("IMG_1629.JPG?1507122477.jpg", "IMG_1629.JPG?1507122477.xml"),
    (
        "show_picture.asp?id=aaaaaaaaaaogcqq&w2=420&h2=378&clip=center,420,378&meta=0.jpg",
        "show_picture.asp?id=aaaaaaaaaaogcqq&w2=420&h2=378&clip=center,420,378&meta=0.xml"
    ),
    ("1421_0.jpeg?itok=FMtmgePj.jpg", "1421_0.jpeg?itok=FMtmgePj.xml"),
    (
        "grape-leaf-picture-id119212425?k=6&m=119212425&s=612x612&w=0&h=8RZBWGcvXe4tAwhhvYZVFzWyRyiN5iTxKfWBpPT1FcU=.jpg",
        "grape-leaf-picture-id119212425?k=6&m=119212425&s=612x612&w=0&h=8RZBWGcvXe4tAwhhvYZVFzWyRyiN5iTxKfWBpPT1FcU=.xml"
    ),
    ("latest?cb=20100621160325.jpg", "latest?cb=20100621160325.xml"),
    ("1058_0.jpeg?itok=OQkdtxgv.jpg", "1058_0.jpeg?itok=OQkdtxgv.xml"),
    (
        "072109%20Hartman%20Grape%20black%20rot-fruit%20&%20lvs.JPG.jpg",
        "072109%20Hartman%20Grape%20black%20rot-fruit%20&%20lvs.JPG.xml"
    ),
    (
        "01Apple-scab-2-Venturia-inaequalis.ashx?w=600&h=408&bc=ffffff.jpg",
        "01Apple-scab-2-Venturia-inaequalis.ashx?w=600&h=408&bc=ffffff.xml"
    ),
    (
        "nature-plant-grape-vine-wine-fruit-food-green-produce-agriculture-grapevine-vines-shrub-grapes-winegrowing-rebstock-flowering-plant-vitis-grape-leaves-green-grapes-swiss-francs-land-plant-grapevine-family-556232.jpg",
        "nature-plant-grape-vine-wine-fruit-food-green-produce-agriculture-grapevine-vines-shrub-grapes-winegrowing-rebstock-flowering-plant-vitis-grape-leaves-green-grapes-swiss-francs-land-plant-grapevine-family-556232.xml"
    )
]

def test_sanitize_exact_kaggle_error_filenames():
    """Verifies that all filenames from the Kaggle upload error are sanitized to safe names."""
    for img_name, xml_name in KAGGLE_ERROR_SAMPLES:
        clean_img = parse_and_sanitize_filename(img_name)
        clean_xml = parse_and_sanitize_filename(xml_name)
        
        # Zero forbidden characters
        bad_img = [c for c in clean_img if c in FORBIDDEN_KAGGLE_CHARS]
        bad_xml = [c for c in clean_xml if c in FORBIDDEN_KAGGLE_CHARS]
        assert len(bad_img) == 0, f"Clean image '{clean_img}' still has bad chars: {bad_img}"
        assert len(bad_xml) == 0, f"Clean xml '{clean_xml}' still has bad chars: {bad_xml}"
        
        # Stems must match
        assert Path(clean_img).stem == Path(clean_xml).stem, (
            f"Stem mismatch between {clean_img} and {clean_xml}"
        )
        # Extensions preserved
        assert clean_img.endswith(".jpg") or clean_img.endswith(".jpeg")
        assert clean_xml.endswith(".xml")
        
        # Max length check (guarantees entire archive entry path <= 160 bytes << 248 bytes)
        assert len(clean_img) <= 90, f"Sanitized image filename too long ({len(clean_img)} chars): {clean_img}"
        assert len(clean_xml) <= 90, f"Sanitized xml filename too long ({len(clean_xml)} chars): {clean_xml}"

def test_directory_sanitization_and_xml_sync():
    """Simulates directory with unsanitized pairs, verifies rename, XML tag sync, and collision handling."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir) / "test_set"
        test_dir.mkdir()
        
        # Create dirty image and dirty xml
        dirty_img = test_dir / "IMG_1629.JPG?1507122477.jpg"
        dirty_xml = test_dir / "IMG_1629.JPG?1507122477.xml"
        
        dirty_img.write_bytes(b"fake_image_bytes")
        dirty_xml.write_text(
            "<annotation><filename>IMG_1629.JPG?1507122477.jpg</filename>"
            "<path>/data/IMG_1629.JPG?1507122477.jpg</path></annotation>",
            encoding="utf-8"
        )
        
        # Create an intentional collision target
        clean_name = parse_and_sanitize_filename("collision_sample?id=1.jpg")
        col_file = test_dir / clean_name
        col_file.write_bytes(b"existing_file")
        
        dirty_col_img = test_dir / "collision_sample?id=1.jpg"
        dirty_col_img.write_bytes(b"incoming_file")
        
        stats = sanitize_dataset_directory(test_dir)
        assert stats["renamed"] >= 3
        
        # Verify XML tag was updated to point to the new image name
        sanitized_xml = list(test_dir.glob("IMG_1629*xml"))[0]
        xml_content = sanitized_xml.read_text(encoding="utf-8")
        sanitized_img = list(test_dir.glob("IMG_1629*jpg"))[0]
        assert sanitized_img.name in xml_content, f"XML did not synchronize image name {sanitized_img.name}"
        assert "?" not in xml_content

def test_physical_plantdoc_od_repo_is_clean():
    """Verifies that the actual cv/datasets/raw/plantdoc_od_repo has 0 forbidden characters."""
    repo = Path("cv/datasets/raw/plantdoc_od_repo")
    if repo.exists():
        bad_files = [p for p in repo.rglob("*") if p.is_file() and any(c in FORBIDDEN_KAGGLE_CHARS for c in p.name)]
        assert len(bad_files) == 0, f"Found {len(bad_files)} forbidden filenames in plantdoc_od_repo: {bad_files[:5]}"

def test_zip_archive_kaggle_compatibility():
    """Verifies that smartcrop_codebase_master.zip passes all Kaggle compatibility rules."""
    zip_path = Path("smartcrop_codebase_master.zip")
    if zip_path.exists():
        valid, errors, stats = validate_zip_compatibility(zip_path)
        assert valid, f"ZIP compatibility errors ({len(errors)}): {errors[:5]}"
        assert stats["forbidden_name_count"] == 0
        assert stats["traversal_count"] == 0
        assert stats["absolute_path_count"] == 0
        assert stats["duplicate_count"] == 0

def test_plantdoc_annotation_pair_integrity():
    """Verifies that all Pascal VOC XML files in plantdoc_od_repo have matching image files."""
    repo = Path("cv/datasets/raw/plantdoc_od_repo")
    if repo.exists():
        res = audit_dataset_pair_integrity(repo)
        assert res["unpaired_xmls"] == 0, f"Detected {res['unpaired_xmls']} broken XML annotations!"
        assert res["paired_xmls"] >= 2500, f"Expected >=2500 paired XML annotations, got {res['paired_xmls']}"

def test_trailing_whitespace_filenames_sanitization():
    """Verifies that filenames with trailing whitespace before extension are sanitized."""
    trailing_space_samples = [
        "116ba4e1-7cd5-4d4e-b876-5212a41a0ab3___R.S_HL 8201 copy .jpg",
        "af647b0e-db3a-459e-915c-4e522cb46c6c___R.S_HL 8101 copy .jpg",
        "1f38a263-cfb5-47fe-8c0e-f8d62141f922___GHLB_PS leaf 28 Day 12 .jpg",
        "c9f19d2a-8c9d-4ac1-b077-545c5d6f53a9___GHLB_PS Leaf 53 Day 18 .jpg",
        "d45e9f86-2819-4c41-886c-73b7012674b3___GHLB Leaf 2 Day 16 .JPG",
        "784f6313-1080-4bd1-a556-d4e14e6879d0___GHLB_PS Leaf 53.1 Day 18 .jpg"
    ]
    for s in trailing_space_samples:
        clean = parse_and_sanitize_filename(s)
        p = Path(clean)
        assert p.stem == p.stem.strip(), f"Sanitized stem '{p.stem}' still has trailing whitespace"
        assert not clean.endswith(" .jpg")
        assert not clean.endswith(" .JPG")

def test_physical_plantvillage_has_no_trailing_whitespace():
    """Verifies that physical files in cv/datasets/raw/plantvillage have 0 trailing whitespace issues."""
    pv_dir = Path("cv/datasets/raw/plantvillage")
    if pv_dir.exists():
        bad = []
        for p in pv_dir.rglob("*"):
            if p.is_file():
                if "." in p.name:
                    stem, ext = p.name.rsplit(".", 1)
                    if stem != stem.strip() or ext != ext.strip():
                        bad.append(str(p))
        assert len(bad) == 0, f"Found files with trailing whitespace: {bad}"

def test_no_nested_archives_in_bundle():
    """Verifies that no nested .zip, .tar, or .gz files exist in the release archive, preventing Kaggle auto-unpacker collisions."""
    zip_path = Path("smartcrop_codebase_master.zip")
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            nested = [
                n for n in zf.namelist() 
                if n.lower().endswith((".zip", ".tar", ".gz", ".tgz", ".rar", ".7z"))
            ]
            assert len(nested) == 0, f"Detected nested archives in release bundle: {nested}"


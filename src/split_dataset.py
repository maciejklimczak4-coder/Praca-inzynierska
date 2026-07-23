import csv
from collections import defaultdict
from pathlib import Path
from sklearn.model_selection import train_test_split
import cv2
import shutil

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

def get_samples(root):
    samples = []
    for s in (root/"train"/"good").iterdir():
        if s.suffix.lower() in VALID_EXTENSIONS:
            samples.append({"filepath": s, "category": "OK", "defect_type": "good"})
    for subfolder in (root/"test").iterdir():
        category = "OK" if subfolder.name == "good" else "NOK"
        for s in subfolder.iterdir():
            if s.suffix.lower() in VALID_EXTENSIONS:
                samples.append({"filepath": s, "category": category, "defect_type": subfolder.name})
    return samples

def unique_stem(sample):
    return f"{sample['defect_type']}__{sample['filepath'].stem}"

root = Path(__file__).resolve().parent.parent/"data"
samples = get_samples(root)

train, rest = train_test_split(samples, test_size=0.3, stratify=[s['category']for s in samples], random_state=42)
val, test = train_test_split(rest, test_size=0.5, stratify=[s['category']for s in rest], random_state=42)

split_lookup = {}
for split_name, split_samples in (("train", train),("val", val),("test", test)):
    for s in split_samples:
        relative_path = str(s['filepath'].relative_to(root))
        split_lookup[relative_path] = split_name

with open(root/"split.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filepath", "category", "defect_type", "split"])
    for s in samples:
        relative_path = str(s['filepath'].relative_to(root))
        writer.writerow([relative_path, s['category'], s['defect_type'], split_lookup[relative_path]])

# Zmiana bounding boxów na format YOLO, żeby później wytrenowac model
bboxes_by_image = defaultdict(list)
for defect_dir in (root/"ground_truth").iterdir():
    defect_type = defect_dir.name
    for mask_path in defect_dir.iterdir():
        image_stem = mask_path.stem.removesuffix("_mask")
        candidates = list((root/"test"/defect_type).glob(f"{image_stem}.*"))
        if not candidates:
            print(f"No picture for mask {mask_path}, skip.")
            continue
        image_path = candidates[0]
        relative_image_path = str(image_path.relative_to(root))

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Could not load mask {mask_path}, skip.")
            continue

        img_h, img_w = mask.shape
        _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            x_center = (x + w/2)/ img_w
            y_center = (y + h/2)/ img_h
            bboxes_by_image[relative_image_path].append(
                f"0 {x_center:.6f} {y_center:.6f} {w/img_w:.6f} {h/img_h:.6f}"
            )

#Tworzenie etykiet, czyli plików txt dla każdego zdjęcia, po samples żeby nie omijać zdjęć bez wad
for s in samples:
    relative_image_path = str(s['filepath'].relative_to(root))
    split_name = split_lookup[relative_image_path]
    label_dir = root/"labels"/split_name
    label_dir.mkdir(parents=True, exist_ok=True)
    lines = bboxes_by_image.get(relative_image_path,[])
    (label_dir/f"{unique_stem(s)}.txt").write_text("\n".join(lines), encoding="utf-8")

for s in samples:
    relative_image_path = str(s['filepath'].relative_to(root))
    split_name = split_lookup[relative_image_path]
    img_dir = root/"images"/split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    destination = img_dir/f"{unique_stem(s)}{s['filepath'].suffix}"
    if not destination.exists():
        shutil.copy2(s['filepath'].resolve(), destination)

print(f"Split: {len(train)} train / {len(val)} val / {len(test)} test")
print(f"Labels and images saved in {root/'labels'} and {root/'images'}")


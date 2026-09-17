from pathlib import Path

dataset = Path("YOLO_dataset")

errors = []

for split in ["train", "val"]:
    image_dir = dataset / "images" / split
    label_dir = dataset / "labels" / split

    images = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg"))
    labels = list(label_dir.glob("*.txt"))

    image_stems = {img.stem for img in images}
    label_stems = {lbl.stem for lbl in labels}

    # Check missing labels
    for stem in image_stems - label_stems:
        errors.append(f"{split}: Missing label for {stem}")

    # Check labels without images
    for stem in label_stems - image_stems:
        errors.append(f"{split}: Label without image: {stem}")

    # Check label contents
    for label_file in labels:
        with open(label_file, "r") as f:
            lines = f.readlines()

        for line_no, line in enumerate(lines, start=1):
            values = line.strip().split()

            if len(values) != 5:
                errors.append(
                    f"{label_file}: line {line_no} has {len(values)} values"
                )
                continue

            try:
                class_id = int(values[0])
                coords = list(map(float, values[1:]))
            except ValueError:
                errors.append(
                    f"{label_file}: line {line_no} contains invalid values"
                )
                continue

            if class_id != 0:
                errors.append(
                    f"{label_file}: line {line_no} has class {class_id}"
                )

            if not all(0 <= x <= 1 for x in coords):
                errors.append(
                    f"{label_file}: line {line_no} has coordinates outside 0-1"
                )

    print(f"\n{split.upper()} DATASET")
    print(f"Images : {len(images)}")
    print(f"Labels : {len(labels)}")


print("\n" + "=" * 50)

if errors:
    print(f"FOUND {len(errors)} PROBLEM(S):\n")
    for error in errors[:50]:
        print(error)
else:
    print("DATASET VALIDATION PASSED!")
    print("No missing files or invalid YOLO annotations found.")

print("=" * 50)
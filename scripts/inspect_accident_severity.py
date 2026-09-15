"""
Comprehensive Dataset Inspection for Accident Severity Dataset.

Inspects datasets/accident_severity/ for:
- Dataset split statistics
- Class distribution
- Dataset integrity
- Image quality
- Bounding box quality
- Class semantics
"""

import cv2
from pathlib import Path
from collections import defaultdict
import json


def count_class_annotations(label_dir: Path) -> dict:
    """Count annotations per class from label files."""
    class_counts = defaultdict(int)
    image_counts = defaultdict(int)
    
    for label_file in label_dir.glob("*.txt"):
        classes_in_image = set()
        
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
                    classes_in_image.add(class_id)
        
        for class_id in classes_in_image:
            image_counts[class_id] += 1
    
    return dict(class_counts), dict(image_counts)


def check_integrity(split_name: str, data_dir: Path) -> dict:
    """Check dataset integrity for missing files, empty labels, invalid formats."""
    images_dir = data_dir / split_name / "images"
    labels_dir = data_dir / split_name / "labels"
    
    issues = {
        'missing_images': [],
        'missing_labels': [],
        'empty_labels': [],
        'invalid_class_ids': [],
        'invalid_format': []
    }
    
    image_files = {f.stem: f for f in images_dir.glob("*.jpg")}
    label_files = {f.stem: f for f in labels_dir.glob("*.txt")}
    
    # Check for missing images
    for label_stem in label_files:
        if label_stem not in image_files:
            issues['missing_images'].append(label_stem)
    
    # Check for missing labels
    for image_stem in image_files:
        if image_stem not in label_files:
            issues['missing_labels'].append(image_stem)
    
    # Check label files
    for label_file in labels_dir.glob("*.txt"):
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        if len(lines) == 0:
            issues['empty_labels'].append(label_file.name)
            continue
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                issues['invalid_format'].append(f"{label_file.name}: {line.strip()}")
                continue
            
            try:
                class_id = int(parts[0])
                if class_id not in [0, 1, 2]:
                    issues['invalid_class_ids'].append(f"{label_file.name}: class {class_id}")
            except ValueError:
                issues['invalid_format'].append(f"{label_file.name}: {line.strip()}")
    
    return issues


def analyze_image_quality(images_dir: Path, sample_size: int = 100) -> dict:
    """Analyze image quality and resolutions."""
    image_files = list(images_dir.glob("*.jpg"))
    resolutions = []
    sample_files = image_files[:sample_size] if len(image_files) > sample_size else image_files
    
    for img_file in sample_files:
        img = cv2.imread(str(img_file))
        if img is not None:
            h, w = img.shape[:2]
            resolutions.append((w, h))
    
    if resolutions:
        widths = [r[0] for r in resolutions]
        heights = [r[1] for r in resolutions]
        
        return {
            'sample_count': len(resolutions),
            'min_resolution': (min(widths), min(heights)),
            'max_resolution': (max(widths), max(heights)),
            'common_resolutions': defaultdict(int)
        }
    
    return {'sample_count': 0}


def check_bounding_boxes(labels_dir: Path, sample_size: int = 100) -> dict:
    """Check bounding box quality."""
    label_files = list(labels_dir.glob("*.txt"))
    sample_files = label_files[:sample_size] if len(label_files) > sample_size else label_files
    
    issues = {
        'extremely_small': [],
        'extremely_large': [],
        'outside_bounds': []
    }
    
    for label_file in sample_files:
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Check for extremely small boxes (< 1% of image)
                        if width < 0.01 or height < 0.01:
                            issues['extremely_small'].append(label_file.name)
                        
                        # Check for extremely large boxes (> 95% of image)
                        if width > 0.95 or height > 0.95:
                            issues['extremely_large'].append(label_file.name)
                        
                        # Check if box is outside bounds
                        if (x_center < 0 or x_center > 1 or 
                            y_center < 0 or y_center > 1 or
                            x_center - width/2 < 0 or x_center + width/2 > 1 or
                            y_center - height/2 < 0 or y_center + height/2 > 1):
                            issues['outside_bounds'].append(label_file.name)
                    except ValueError:
                        pass
    
    return issues


def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "datasets" / "accident_severity"
    
    print("=" * 70)
    print("  ACCIDENT SEVERITY DATASET INSPECTION")
    print("=" * 70)
    
    # Read data.yaml
    yaml_file = data_dir / "data.yaml"
    with open(yaml_file, 'r') as f:
        yaml_content = f.read()
    
    print("\n[1] DATASET SPLIT")
    print("-" * 70)
    
    splits = {}
    for split in ['train', 'valid', 'test']:
        images_dir = data_dir / split / "images"
        labels_dir = data_dir / split / "labels"
        
        image_count = len(list(images_dir.glob("*.jpg")))
        label_count = len(list(labels_dir.glob("*.txt")))
        
        splits[split] = {
            'images': image_count,
            'labels': label_count
        }
        
        print(f"{split.upper()}:")
        print(f"  Images: {image_count}")
        print(f"  Labels: {label_count}")
    
    total_images = sum(s['images'] for s in splits.values())
    print(f"\nTOTAL IMAGES: {total_images}")
    
    print("\n[2] CLASS DISTRIBUTION")
    print("-" * 70)
    
    class_names = ['moderate-accident', 'no-accident', 'severe-accident']
    
    total_annotations = defaultdict(int)
    total_images_per_class = defaultdict(int)
    
    for split in ['train', 'valid', 'test']:
        labels_dir = data_dir / split / "labels"
        class_counts, image_counts = count_class_annotations(labels_dir)
        
        print(f"\n{split.upper()}:")
        for class_id in range(3):
            print(f"  {class_names[class_id]}:")
            print(f"    Images with class: {image_counts.get(class_id, 0)}")
            print(f"    Annotations: {class_counts.get(class_id, 0)}")
            
            total_annotations[class_id] += class_counts.get(class_id, 0)
            total_images_per_class[class_id] += image_counts.get(class_id, 0)
    
    print(f"\nTOTAL ANNOTATIONS:")
    total_ann_sum = sum(total_annotations.values())
    for class_id in range(3):
        count = total_annotations[class_id]
        percentage = (count / total_ann_sum * 100) if total_ann_sum > 0 else 0
        print(f"  {class_names[class_id]}: {count} ({percentage:.2f}%)")
    
    print(f"\nTOTAL IMAGES PER CLASS:")
    total_img_sum = sum(total_images_per_class.values())
    for class_id in range(3):
        count = total_images_per_class[class_id]
        percentage = (count / total_img_sum * 100) if total_img_sum > 0 else 0
        print(f"  {class_names[class_id]}: {count} ({percentage:.2f}%)")
    
    print("\n[3] DATASET INTEGRITY")
    print("-" * 70)
    
    all_issues = {
        'missing_images': [],
        'missing_labels': [],
        'empty_labels': [],
        'invalid_class_ids': [],
        'invalid_format': []
    }
    
    for split in ['train', 'valid', 'test']:
        issues = check_integrity(split, data_dir)
        for key in all_issues:
            all_issues[key].extend(issues[key])
    
    for key, value in all_issues.items():
        count = len(value)
        if count > 0:
            print(f"{key.replace('_', ' ').title()}: {count}")
            if count <= 5:
                for item in value:
                    print(f"  - {item}")
            else:
                print(f"  (showing first 5)")
                for item in value[:5]:
                    print(f"  - {item}")
        else:
            print(f"{key.replace('_', ' ').title()}: 0 ✓")
    
    print("\n[4] IMAGE QUALITY")
    print("-" * 70)
    
    train_images_dir = data_dir / "train" / "images"
    quality_stats = analyze_image_quality(train_images_dir, sample_size=200)
    
    if quality_stats['sample_count'] > 0:
        print(f"Sample size: {quality_stats['sample_count']} images")
        print(f"Min resolution: {quality_stats['min_resolution'][0]}x{quality_stats['min_resolution'][1]}")
        print(f"Max resolution: {quality_stats['max_resolution'][0]}x{quality_stats['max_resolution'][1]}")
    else:
        print("Could not analyze image quality")
    
    print("\n[5] BOUNDING BOX QUALITY")
    print("-" * 70)
    
    train_labels_dir = data_dir / "train" / "labels"
    bbox_issues = check_bounding_boxes(train_labels_dir, sample_size=200)
    
    for key, value in bbox_issues.items():
        count = len(set(value))  # Unique files
        if count > 0:
            print(f"{key.replace('_', ' ').title()}: {count} files")
        else:
            print(f"{key.replace('_', ' ').title()}: 0 ✓")
    
    print("\n[6] CLASS SEMANTICS")
    print("-" * 70)
    print("Visual inspection of representative images:")
    print("  - Requires manual review of sample images from each class")
    print("  - Check if visual content matches class labels:")
    print("    * moderate-accident: Should show moderate damage/impact")
    print("    * no-accident: Should show normal traffic/no accident")
    print("    * severe-accident: Should show severe damage/impact")
    
    print("\n[7] FINAL RECOMMENDATION")
    print("-" * 70)
    
    critical_issues = (
        len(all_issues['missing_images']) > 0 or
        len(all_issues['missing_labels']) > 0 or
        len(all_issues['invalid_class_ids']) > 0
    )
    
    if critical_issues:
        print("NEEDS CLEANING")
        print("\nCritical issues found:")
        if all_issues['missing_images']:
            print(f"  - {len(all_issues['missing_images'])} missing image files")
        if all_issues['missing_labels']:
            print(f"  - {len(all_issues['missing_labels'])} missing label files")
        if all_issues['invalid_class_ids']:
            print(f"  - {len(all_issues['invalid_class_ids'])} invalid class IDs")
    else:
        print("READY FOR TRAINING")
        print("\nNo critical integrity issues found.")
    
    # Save report
    output_dir = base_dir / "results" / "detection" / "accident_severity"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = output_dir / "dataset_inspection.txt"
    
    with open(report_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("  ACCIDENT SEVERITY DATASET INSPECTION REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("DATASET PATH: " + str(data_dir) + "\n\n")
        
        f.write("[1] DATASET SPLIT\n")
        f.write("-" * 70 + "\n")
        for split in ['train', 'valid', 'test']:
            f.write(f"{split.upper()}:\n")
            f.write(f"  Images: {splits[split]['images']}\n")
            f.write(f"  Labels: {splits[split]['labels']}\n\n")
        f.write(f"TOTAL IMAGES: {total_images}\n\n")
        
        f.write("[2] CLASS DISTRIBUTION\n")
        f.write("-" * 70 + "\n")
        f.write("TOTAL ANNOTATIONS:\n")
        for class_id in range(3):
            count = total_annotations[class_id]
            percentage = (count / total_ann_sum * 100) if total_ann_sum > 0 else 0
            f.write(f"  {class_names[class_id]}: {count} ({percentage:.2f}%)\n")
        
        f.write("\nTOTAL IMAGES PER CLASS:\n")
        for class_id in range(3):
            count = total_images_per_class[class_id]
            percentage = (count / total_img_sum * 100) if total_img_sum > 0 else 0
            f.write(f"  {class_names[class_id]}: {count} ({percentage:.2f}%)\n")
        
        f.write("\n[3] DATASET INTEGRITY\n")
        f.write("-" * 70 + "\n")
        for key, value in all_issues.items():
            count = len(value)
            f.write(f"{key.replace('_', ' ').title()}: {count}\n")
        
        f.write("\n[4] IMAGE QUALITY\n")
        f.write("-" * 70 + "\n")
        if quality_stats['sample_count'] > 0:
            f.write(f"Sample size: {quality_stats['sample_count']} images\n")
            f.write(f"Min resolution: {quality_stats['min_resolution'][0]}x{quality_stats['min_resolution'][1]}\n")
            f.write(f"Max resolution: {quality_stats['max_resolution'][0]}x{quality_stats['max_resolution'][1]}\n")
        
        f.write("\n[5] BOUNDING BOX QUALITY\n")
        f.write("-" * 70 + "\n")
        for key, value in bbox_issues.items():
            count = len(set(value))
            f.write(f"{key.replace('_', ' ').title()}: {count} files\n")
        
        f.write("\n[6] DATA.YAML CONTENT\n")
        f.write("-" * 70 + "\n")
        f.write(yaml_content)
        
        f.write("\n[7] FINAL RECOMMENDATION\n")
        f.write("-" * 70 + "\n")
        if critical_issues:
            f.write("NEEDS CLEANING\n")
        else:
            f.write("READY FOR TRAINING\n")
    
    print(f"\n[+] Report saved to: {report_file}")
    
    print("\n" + "=" * 70)
    print("  INSPECTION SUMMARY")
    print("=" * 70)
    print(f"Total images: {total_images}")
    print(f"Train: {splits['train']['images']}, Valid: {splits['valid']['images']}, Test: {splits['test']['images']}")
    print(f"\nClass distribution:")
    for class_id in range(3):
        count = total_annotations[class_id]
        percentage = (count / total_ann_sum * 100) if total_ann_sum > 0 else 0
        print(f"  {class_names[class_id]}: {count} ({percentage:.2f}%)")
    
    if critical_issues:
        print(f"\nSerious problems: YES")
        print(f"Recommendation: NEEDS CLEANING")
    else:
        print(f"\nSerious problems: NO")
        print(f"Recommendation: READY FOR TRAINING")
    
    print("=" * 70)


if __name__ == "__main__":
    main()

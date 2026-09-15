"""
Filter Night Vision Dataset for Traffic Classes.

Filters the night_vision dataset to keep only traffic-related classes:
- Bicycle (original class 0 -> new class 0)
- Bus (original class 3 -> new class 1)
- Car (original class 4 -> new class 2)
- Motorbike (original class 9 -> new class 3)
- People (original class 10 -> new class 4)

Source: datasets/night_vision/
Target: datasets/night_traffic/
"""

import shutil
from pathlib import Path
from collections import defaultdict
import yaml


# Original class IDs to keep and their new mappings
CLASS_MAPPING = {
    0: 0,   # Bicycle -> Bicycle
    3: 1,   # Bus -> Bus
    4: 2,   # Car -> Car
    9: 3,   # Motorbike -> Motorbike
    10: 4   # People -> People
}

CLASS_NAMES = ['Bicycle', 'Bus', 'Car', 'Motorbike', 'People']


def filter_split(split_name: str, source_dir: Path, target_dir: Path, stats: dict):
    """
    Filter a single split (train/valid/test).
    
    Args:
        split_name: Name of the split ('train', 'valid', 'test')
        source_dir: Source dataset directory
        target_dir: Target dataset directory
        stats: Dictionary to collect statistics
    """
    print(f"\n[+] Processing {split_name} split...")
    
    source_images_dir = source_dir / split_name / "images"
    source_labels_dir = source_dir / split_name / "labels"
    
    target_images_dir = target_dir / split_name / "images"
    target_labels_dir = target_dir / split_name / "labels"
    
    target_images_dir.mkdir(parents=True, exist_ok=True)
    target_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all label files
    label_files = list(source_labels_dir.glob("*.txt"))
    
    print(f"    Found {len(label_files)} label files")
    
    images_copied = 0
    class_counts = defaultdict(int)
    
    for label_file in label_files:
        # Read label file
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        # Filter and remap annotations
        filtered_annotations = []
        has_valid_annotation = False
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            original_class_id = int(parts[0])
            
            # Check if this class should be kept
            if original_class_id in CLASS_MAPPING:
                new_class_id = CLASS_MAPPING[original_class_id]
                # Remap the class ID
                parts[0] = str(new_class_id)
                filtered_annotations.append(' '.join(parts))
                class_counts[new_class_id] += 1
                has_valid_annotation = True
        
        # Only copy if there are valid annotations
        if has_valid_annotation:
            # Copy image
            image_file = source_images_dir / (label_file.stem + ".jpg")
            if image_file.exists():
                shutil.copy2(image_file, target_images_dir / image_file.name)
                images_copied += 1
            
            # Write filtered label file
            target_label_file = target_labels_dir / label_file.name
            with open(target_label_file, 'w') as f:
                f.write('\n'.join(filtered_annotations))
    
    print(f"    Images copied: {images_copied}")
    print(f"    Class distribution:")
    for class_id in range(5):
        print(f"      {CLASS_NAMES[class_id]}: {class_counts[class_id]}")
    
    stats[split_name] = {
        'images': images_copied,
        'class_counts': dict(class_counts)
    }


def create_data_yaml(target_dir: Path):
    """Create data.yaml for the filtered dataset."""
    data_config = {
        'train': '../night_traffic/train/images',
        'val': '../night_traffic/valid/images',
        'test': '../night_traffic/test/images',
        'nc': 5,
        'names': CLASS_NAMES
    }
    
    yaml_path = target_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n[+] Created data.yaml at: {yaml_path}")


def main():
    base_dir = Path(__file__).resolve().parent.parent
    source_dir = base_dir / "datasets" / "night_vision"
    target_dir = base_dir / "datasets" / "night_traffic"
    
    print("=" * 70)
    print("  NIGHT VISION DATASET FILTERING")
    print("=" * 70)
    print(f"\nSource: {source_dir}")
    print(f"Target: {target_dir}")
    print(f"\nClasses to keep: {CLASS_NAMES}")
    print(f"Class mapping: {CLASS_MAPPING}")
    
    # Verify source exists
    if not source_dir.exists():
        print(f"[!] Source directory not found: {source_dir}")
        return
    
    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Statistics
    stats = {}
    
    # Process each split
    for split in ['train', 'valid', 'test']:
        filter_split(split, source_dir, target_dir, stats)
    
    # Create data.yaml
    create_data_yaml(target_dir)
    
    # Print summary
    print("\n" + "=" * 70)
    print("  FILTERING SUMMARY")
    print("=" * 70)
    
    total_images = 0
    total_annotations = defaultdict(int)
    
    for split in ['train', 'valid', 'test']:
        if split in stats:
            print(f"\n{split.upper()}:")
            print(f"  Images: {stats[split]['images']}")
            total_images += stats[split]['images']
            
            for class_id in range(5):
                count = stats[split]['class_counts'].get(class_id, 0)
                total_annotations[class_id] += count
                print(f"  {CLASS_NAMES[class_id]}: {count}")
    
    print(f"\nTOTAL:")
    print(f"  Images: {total_images}")
    for class_id in range(5):
        print(f"  {CLASS_NAMES[class_id]}: {total_annotations[class_id]}")
    
    print("\n" + "=" * 70)
    print("[+] Filtering complete!")
    print(f"[+] Original dataset not modified: {source_dir}")
    print(f"[+] New dataset created at: {target_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

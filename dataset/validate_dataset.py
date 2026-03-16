import os
import cv2
import yaml
from collections import defaultdict

def validate_dataset(dataset_yaml_path):
    print("--- Starting Dataset Validation ---")
    
    if not os.path.exists(dataset_yaml_path):
        print(f"Error: {dataset_yaml_path} not found.")
        return

    with open(dataset_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)

    classes = data_config.get('names', [])
    dataset_path = data_config.get('path', os.path.dirname(dataset_yaml_path))
    
    splits = ['train', 'val', 'test']
    
    class_counts = defaultdict(int)
    total_images = 0
    missing_labels = 0
    invalid_bboxes = 0
    corrupted_images = 0
    
    for split in splits:
        # Resolve paths based on typical YOLO structure or relative paths in yaml
        img_dir_rel = data_config.get(split, f"images/{split}")
        img_dir_abs = os.path.join(dataset_path, img_dir_rel) if not os.path.isabs(img_dir_rel) else img_dir_rel
        lbl_dir_abs = img_dir_abs.replace('images', 'labels')

        if not os.path.exists(img_dir_abs):
            print(f"Warning: Split '{split}' image directory not found: {img_dir_abs}")
            continue
            
        print(f"Validating split: {split}")
        
        for root, _, files in os.walk(img_dir_abs):
            for file in files:
                if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    continue
                
                total_images += 1
                img_path = os.path.join(root, file)
                
                # Check for corrupted image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Corrupted image detected: {img_path}")
                    corrupted_images += 1
                    continue
                
                # Check label file
                rel_path = os.path.relpath(root, img_dir_abs)
                lbl_root = os.path.normpath(os.path.join(lbl_dir_abs, rel_path))
                lbl_file = os.path.splitext(file)[0] + '.txt'
                lbl_path = os.path.join(lbl_root, lbl_file)
                
                if not os.path.exists(lbl_path):
                    missing_labels += 1
                    continue
                
                # Validate bounding boxes and count classes
                with open(lbl_path, 'r') as lf:
                    for line_num, line in enumerate(lf):
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls_id, x, y, w, h = map(float, parts)
                            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                                invalid_bboxes += 1
                                print(f"Invalid bbox in {lbl_path} (line {line_num+1})")
                            class_counts[int(cls_id)] += 1
                        else:
                            print(f"Malformed label line in {lbl_path} (line {line_num+1})")
                            
    # Generate Summary Report
    print("\n--- Dataset Validation Summary ---")
    print(f"Total Images Validated: {total_images}")
    print(f"Corrupted/Unreadable Images: {corrupted_images}")
    print(f"Images Missing Labels: {missing_labels}")
    print(f"Invalid Bounding Boxes Detected: {invalid_bboxes}")
    
    print("\nClass Distribution:")
    for cls_id in range(len(classes)):
        name = classes[cls_id] if isinstance(classes, list) else classes[cls_id]
        print(f"  Class {cls_id} ({name}): {class_counts[cls_id]} instances")
        
    print("----------------------------------\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate YOLO dataset.")
    parser.add_argument('--yaml', type=str, default='dataset/dataset.yaml', help='Path to dataset.yaml')
    args = parser.parse_args()
    
    validate_dataset(args.yaml)

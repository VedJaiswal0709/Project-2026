import os
import glob
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define target classes and their IDs
TARGET_CLASSES = {
    'car': 0,
    'bus': 1,
    'truck': 2,
    'motorcycle': 3,
    'bicycle': 4,
    'auto_rickshaw': 5,
    'e_rickshaw': 6,
    'magic_vehicle': 7
}

def convert_bbox_to_yolo(img_width, img_height, xmin, ymin, xmax, ymax):
    """
    Convert boundary box coordinates (xmin, ymin, xmax, ymax) to YOLO format (x_center, y_center, width, height)
    Coordinates should be raw pixel values.
    Returns normalized values [0, 1].
    """
    x_center = (xmin + xmax) / 2.0 / img_width
    y_center = (ymin + ymax) / 2.0 / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    return x_center, y_center, width, height

def format_yolo_line(class_id, x_center, y_center, width, height):
    """
    Format a single annotation line in YOLO format.
    """
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"

def validate_yolo_file(label_file_path):
    """
    Validates a single YOLO format label file.
    Ensures file follows `class_id x_center y_center width height`
    and that bounding box coordinates are normalized (0 to 1).
    """
    is_valid = True
    with open(label_file_path, 'r') as f:
        lines = f.readlines()
        
    for idx, line in enumerate(lines):
        parts = line.strip().split()
        if not parts:
            continue
            
        if len(parts) != 5:
            logging.error(f"File {label_file_path} (Line {idx+1}): Incorrect number of parameters. Expected 5, got {len(parts)}.")
            is_valid = False
            continue
            
        class_id = int(parts[0])
        if class_id not in TARGET_CLASSES.values():
            logging.error(f"File {label_file_path} (Line {idx+1}): Invalid class_id '{class_id}'.")
            is_valid = False
            
        try:
            x_center, y_center, width, height = map(float, parts[1:5])
            if any(val < 0.0 or val > 1.0 for val in [x_center, y_center, width, height]):
                logging.error(f"File {label_file_path} (Line {idx+1}): Coordinates out of bounds (0-1).")
                is_valid = False
        except ValueError:
             logging.error(f"File {label_file_path} (Line {idx+1}): Coordinates must be floats.")
             is_valid = False

    return is_valid

def verify_dataset_integrity(dataset_dir):
    """
    1. Checks for missing labels.
    2. Verifies that every image has a corresponding label file.
    3. Validates the contents of existing label files.
    """
    logging.info(f"Starting integrity check on dataset: {dataset_dir}")
    
    splits = ['train', 'val', 'test']
    total_images = 0
    total_valid_labels = 0
    missing_labels_count = 0
    
    for split in splits:
        img_dir = os.path.join(dataset_dir, 'images', split)
        label_dir = os.path.join(dataset_dir, 'labels', split)
        
        if not os.path.exists(img_dir) or not os.path.exists(label_dir):
            logging.warning(f"Split '{split}' directory not found. Skipping.")
            continue
            
        images = glob.glob(os.path.join(img_dir, '*.*')) # Check all typical image formats
        
        for img_path in images:
            if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            total_images += 1
            filename = os.path.basename(img_path)
            basename, _ = os.path.splitext(filename)
            label_path = os.path.join(label_dir, f"{basename}.txt")
            
            # Check if corresponding label exists
            if not os.path.exists(label_path):
                logging.error(f"Missing label file for image: {img_path}")
                missing_labels_count += 1
                continue
            
            # Check if label is empty
            if os.path.getsize(label_path) == 0:
                logging.warning(f"Empty label file for image: {img_path}")
                # Technically valid per YOLO (background image), but flag it
                pass 
                
            # Validate contents
            if validate_yolo_file(label_path):
                total_valid_labels += 1
                
    logging.info("--- Integrity Check Summary ---")
    logging.info(f"Total Images: {total_images}")
    logging.info(f"Total Valid Labels: {total_valid_labels}")
    logging.info(f"Missing Labels: {missing_labels_count}")
    
    if total_images == total_valid_labels and missing_labels_count == 0:
         logging.info("Dataset integrity verification PASSED.")
    else:
         logging.error("Dataset integrity verification FAILED. Please review the errors above.")

def generate_dataset_yaml(output_path, dataset_root_path):
    """
    Automatically generates the dataset.yaml file required for YOLO training.
    """
    logging.info(f"Generating dataset.yaml at {output_path}")
    
    # Sort class names based on ID to ensure correct ordering in YAML
    sorted_classes = sorted(TARGET_CLASSES.items(), key=lambda x: x[1])
    class_names = [cls[0] for cls in sorted_classes]
    
    dataset_dict = {
        'path': dataset_root_path, # usually relative or absolute path to dataset root
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(class_names),
        'names': class_names
    }
    
    try:
        with open(output_path, 'w') as f:
            yaml.dump(dataset_dict, f, default_flow_style=False, sort_keys=False)
        logging.info("Successfully generated dataset.yaml")
    except Exception as e:
        logging.error(f"Failed to write yaml file: {e}")

if __name__ == "__main__":
    # Example usage:
    # DATASET_DIR = "dataset_root_directory"
    
    # 1. Verification
    # verify_dataset_integrity(DATASET_DIR)
    
    # 2. YAML Generation
    # generate_dataset_yaml(os.path.join(DATASET_DIR, "dataset.yaml"), DATASET_DIR)
    pass

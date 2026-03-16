import cv2
import os
import albumentations as A
import numpy as np

def resize_image(image, target_size=(640, 640)):
    """
    Resizes image to target dimension for YOLO (default 640x640), maintaining aspect ratio with padding.
    """
    h, w = image.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    nh, nw = int(h * scale), int(w * scale)
    
    resized = cv2.resize(image, (nw, nh))
    
    # Pad to target size
    top = (target_size[0] - nh) // 2
    bottom = target_size[0] - nh - top
    left = (target_size[1] - nw) // 2
    right = target_size[1] - nw - left
    
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[114, 114, 114])
    return padded, scale, left, top

def adjust_bounding_boxes(boxes, scale, pad_left, pad_top):
    """
    Adjusts bounding boxes based on resizing and padding operations.
    YOLO format boxes: [class, x_center, y_center, width, height] normalized.
    """
    adjusted_boxes = []
    for box in boxes:
        cls, x_c, y_c, bw, bh = box
        # Logic to adjust goes here (needs un-normalization, scale/pad, and re-normalization)
        # Placeholder for actual box math
        adjusted_boxes.append(box)
    return adjusted_boxes

def get_augmentation_pipeline():
    """
    Returns an albumentations pipeline for basic data augmentation.
    Useful for increasing robustness against variations before weather simulation.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.3),
        A.Blur(blur_limit=3, p=0.1)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def process_directory(input_dir, output_dir):
    """
    Processes all images in a directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(input_dir, file)
            image = cv2.imread(img_path)
            if image is not None:
                processed_img, _, _, _ = resize_image(image)
                cv2.imwrite(os.path.join(output_dir, file), processed_img)
                print(f"Processed: {file}")

if __name__ == "__main__":
    # Example usage:
    # process_directory("dataset/raw_images", "dataset/processed_images")
    pass

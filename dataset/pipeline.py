import os
import cv2
import numpy as np
import shutil
import hashlib
import random
import logging

# Configure logging to print progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_image_hash(image):
    """
    Calculate MD5 hash of an image (numpy array) to detect and remove duplicates.
    """
    return hashlib.md5(image.tobytes()).hexdigest()

def extract_and_process_frames(video_path, output_dir, frame_skip=10, target_size=(640, 640)):
    """
    1. Extracts frames from a traffic video
    2. Resizes images to 640x640
    3. Removes duplicate images using hashing
    4. Automatically renames images
    """
    logging.info(f"Starting frame extraction for video: {video_path}")
    
    if not os.path.exists(video_path):
        logging.error(f"Video file not found: {video_path}")
        return []

    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Failed to open video: {video_path}")
        return []

    frame_count = 0
    saved_count = 0
    seen_hashes = set()
    saved_images = []
    
    # Get base name for renaming
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process every *frame_skip* frames to reduce temporal redundancy
        if frame_count % frame_skip == 0:
            # Resize image to target dimensions (640x640)
            resized_frame = cv2.resize(frame, target_size)
            
            # Check for duplicates
            img_hash = get_image_hash(resized_frame)
            if img_hash not in seen_hashes:
                seen_hashes.add(img_hash)
                
                # Automatically rename image
                output_filename = f"{video_name}_frame_{saved_count:05d}.jpg"
                output_path = os.path.join(output_dir, output_filename)
                
                # Save the image
                cv2.imwrite(output_path, resized_frame)
                saved_images.append(output_filename)
                saved_count += 1
                
                if saved_count % 50 == 0:
                    logging.info(f"Progress: Extracted and saved {saved_count} unique frames...")

        frame_count += 1

    cap.release()
    logging.info(f"Finished extracting {saved_count} unique frames from {video_path}.")
    return saved_images

def create_yolo_structure(base_dir):
    """
    Creates the standardized YOLO directory structure.
    dataset/
     ├── images/
     │    ├── train/
     │    ├── val/
     │    └── test/
     └── labels/
          ├── train/
          ├── val/
          └── test/
    """
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(base_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'labels', split), exist_ok=True)
    logging.info(f"YOLO directory structure verified at: {os.path.abspath(base_dir)}")

def split_dataset(source_images_dir, dataset_root, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Splits the dataset into train, validation, and test sets.
    """
    logging.info("Starting dataset split...")
    create_yolo_structure(dataset_root)
    
    images = [f for f in os.listdir(source_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)
    
    total = len(images)
    if total == 0:
        logging.warning(f"No images found in {source_images_dir} to split.")
        return

    # Calculate split indices
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]
    
    def move_files(img_list, split_name):
        for img in img_list:
            # Move image
            src_img = os.path.join(source_images_dir, img)
            dst_img = os.path.join(dataset_root, 'images', split_name, img)
            shutil.move(src_img, dst_img)
            
            # Create an empty label file to satisfy YOLO requirements (will be populated manually later)
            label = os.path.splitext(img)[0] + '.txt'
            dst_label = os.path.join(dataset_root, 'labels', split_name, label)
            open(dst_label, 'a').close()

    logging.info("Moving files into train/val/test splits...")
    move_files(train_images, 'train')
    move_files(val_images, 'val')
    move_files(test_images, 'test')
    
    logging.info(f"Dataset split completed successfully.")
    logging.info(f"Total images: {total} | Train: {len(train_images)} ({train_ratio*100:.0f}%) | Val: {len(val_images)} ({val_ratio*100:.0f}%) | Test: {len(test_images)} ({test_ratio*100:.0f}%)")

def main():
    # --- Configuration ---
    # Path to the raw video containing traffic footage
    video_source_path = "raw_videos/traffic_video.mp4" 
    
    # Temporary directory to hold extracted frames before splitting
    temp_frames_dir = "temp_extracted_frames"
    
    # Root directory for the dataset (where 'images' and 'labels' folders will be created)
    dataset_root = "." 
    
    # ---------------------
    
    # Uncomment the following block to run the pipeline when you have a raw video ready:
    
    """
    # 1. Extract frames, resize, remove duplicates, and automatically rename
    extract_and_process_frames(video_source_path, temp_frames_dir, frame_skip=15)
    
    # 2. Split dataset into Train (70%), Validation (20%), and Test (10%)
    split_dataset(temp_frames_dir, dataset_root, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
    
    # 3. Clean up the temporary directory
    if os.path.exists(temp_frames_dir):
        shutil.rmtree(temp_frames_dir)
        logging.info("Cleaned up temporary frames directory.")
    """
    logging.info("Pipeline script is ready. Update 'video_source_path' and run main() to start processing.")

if __name__ == "__main__":
    main()

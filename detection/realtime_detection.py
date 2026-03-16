import cv2
import numpy as np
from ultralytics import YOLO
import sys

# Define vehicle classes to track specifically based on the prompt
TRACKED_CLASSES = ['car', 'bus', 'auto_rickshaw', 'truck']

def draw_info_panel(image, counts, fps):
    """
    Draw a semi-transparent info panel on the top-left corner
    to display live vehicle counts and FPS.
    """
    panel_width = 250
    panel_height = 40 + (len(TRACKED_CLASSES) * 30)
    
    # Create overlay for semi-transparency
    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_width, 10 + panel_height), (0, 0, 0), -1)
    
    # Apply overlay with alpha transparency
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    # Draw FPS
    cv2.putText(image, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw Counts
    y_offset = 70
    for cls in TRACKED_CLASSES:
        count = counts.get(cls, 0)
        # Beautify class name for display (e.g. 'auto_rickshaw' -> 'Auto Rickshaw')
        display_name = cls.replace('_', ' ').title()
        text = f"{display_name}s: {count}"
        cv2.putText(image, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30

def process_frame(frame, model, conf_threshold):
    """
    Runs YOLO inference on a frame, draws bounding boxes, labels,
    confidence scores, and counts the detected vehicles.
    """
    # Run YOLOv8 inference
    results = model.predict(frame, conf=conf_threshold, verbose=False)
    
    # Initialize counts for the current frame
    counts = {cls: 0 for cls in TRACKED_CLASSES}
    
    # The results object contains bounding boxes and class IDs
    res = results[0]
    boxes = res.boxes.cpu() # Get boxes on CPU
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        class_name = res.names[cls_id]
        
        # Increment counter if class is one we are tracking
        if class_name in counts:
            counts[class_name] += 1
            
        # Draw bounding box
        # Use different colors for different classes if desired, defaulting to Green here
        color = (0, 255, 0) 
        if class_name == 'car': color = (255, 0, 0)     # Blue
        elif class_name == 'bus': color = (0, 165, 255) # Orange
        elif class_name == 'truck': color = (0, 0, 255) # Red
        elif class_name == 'auto_rickshaw': color = (0, 255, 255) # Yellow
            
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Display vehicle label and detection confidence
        label = f"{class_name.capitalize()} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Draw label background
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
        # Draw label text
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return frame, counts

def run_realtime_detection(model_path='models/best.pt', source=0, conf_threshold=0.4):
    """
    Main loop to read frames from video or webcam, process them, and display in an OpenCV window.
    
    Args:
        model_path (str): Path to YOLOv8 weights file.
        source (int or str): 0 for webcam, or path to MP4 video.
        conf_threshold (float): Minimum confidence threshold for detection.
    """
    print(f"Loading YOLOv8 model from {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure you have trained the model and exported 'best.pt' to the models directory.")
        return

    # Initialize video capture (0 = default webcam, or provide video file path)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}.")
        return

    # Set up OpenCV window
    window_name = "Indian Traffic Vehicle Detection - Real Time"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Initialize FPS calculator
    fps_start_time = cv2.getTickCount()
    fps_frame_count = 0
    fps = 0.0

    print(f"Starting detection on source '{source}'...")
    print("Press 'q' or 'ESC' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or error reading frame.")
            break
            
        fps_frame_count += 1
        
        # Process the frame (YOLO detection + drawing)
        processed_frame, frame_counts = process_frame(frame, model, conf_threshold)
        
        # Calculate current FPS
        if fps_frame_count >= 10: # Update FPS every 10 frames
            fps_end_time = cv2.getTickCount()
            time_elapsed = (fps_end_time - fps_start_time) / cv2.getTickFrequency()
            fps = fps_frame_count / time_elapsed
            
            # Reset FPS counters
            fps_frame_count = 0
            fps_start_time = cv2.getTickCount()

        # Draw the live counter panel
        draw_info_panel(processed_frame, frame_counts, fps)
        
        # Display the result
        cv2.imshow(window_name, processed_frame)
        
        # Check for exit commands (q or ESC)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Detection session ended.")

if __name__ == "__main__":
    # ---------------------------------------------------------------------
    # To run on webcam: use source=0
    # To run on a video file: use source='path/to/traffic_video.mp4'
    # ---------------------------------------------------------------------
    
    # Example: run_realtime_detection(model_path='../models/best.pt', source='test.mp4')
    
    print("Real-time detection module ready. Configure 'source' in main below to test.")
    
    # Uncomment the following to execute:
    # run_realtime_detection(model_path='../models/best.pt', source=0, conf_threshold=0.45)
    pass

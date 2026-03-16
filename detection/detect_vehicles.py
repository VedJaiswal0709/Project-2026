import cv2
import yaml
from ultralytics import YOLO
import argparse
from collections import defaultdict

def get_color_and_text_color(cls_name):
    cls_name = cls_name.lower()
    # 4-Wheelers and heavy vehicles get Crimson (Red/Pinkish) and White text
    if cls_name in ["car", "bus", "truck", "magic-vehicle", "magic_vehicle"]:
        return (60, 20, 220), (255, 255, 255) 
    # 2/3-Wheelers get Green and Black text
    else:
        return (50, 205, 50), (0, 0, 0)

def draw_boxes(frame, results, class_names):
    """Draws bounding boxes and labels on the frame, counting occurrences."""
    counts = defaultdict(int)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            
            label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            # Replace underscore with hyphen to match user image standard
            label_display = label.replace('_', '-')
            counts[label_display] += 1
            
            bg_color, txt_color = get_color_and_text_color(label_display)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, 2)
            
            # Get text dimensions
            text = f"{label_display}"
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            # Draw solid background for text
            cv2.rectangle(frame, (x1, y1 - 30), (x1 + w + 10, y1), bg_color, -1)
            
            # Draw text over background
            cv2.putText(frame, text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, txt_color, 2)
            
    return frame, counts

def get_class_names(yaml_path):
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            return data.get('names', [])
    except Exception:
        return ["car", "bus", "truck", "motorcycle", "bicycle", "auto_rickshaw", "e_rickshaw", "magic_vehicle"]

def detect_vehicles(source, model_path='models/best.pt', yaml_path='dataset/dataset.yaml', save=False):
    model = YOLO(model_path)
    class_names = get_class_names(yaml_path)
    
    # 0 for webcam, otherwise it's a file
    src = 0 if source == 'webcam' else source
    cap = cv2.VideoCapture(src)
    
    if not cap.isOpened():
        print(f"Error opening source: {source}")
        return

    print("--- Starting Real-time Detection ---")
    print("Press 'q' to quit")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Inference
        results = model(frame, verbose=False)
        
        # Draw boxes and get counts
        annotated_frame, counts = draw_boxes(frame, results, class_names)
        
        # Print counts on screen
        y_pos = 30
        cv2.putText(annotated_frame, "Vehicle Counts:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        for cls, count in counts.items():
            y_pos += 30
            cv2.putText(annotated_frame, f"{cls}: {count}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            print(f"- {cls}: {count}", end=" | ")
        print() # New line for this frame
        
        cv2.imshow('Indian Traffic Detection', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("--- Detection Ended ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='webcam', help='Path to video file, image, or "webcam"')
    parser.add_argument('--model', type=str, default='models/best.pt', help='Path to weights')
    args = parser.parse_args()
    
    detect_vehicles(args.source, args.model)

import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
try:
    import albumentations as A
except ImportError:
    print("Albumentations not installed. Run: pip install albumentations")

def add_rain(image):
    # Simplified rain simulation via Albumentations DropShadow/MotionBlur combo or direct drawing
    # Fallback to direct OpenCV drawing for consistent rain streaks
    result = image.copy()
    h, w = result.shape[:2]
    num_drops = int(0.05 * h * w)
    
    xs = np.random.randint(0, w, num_drops)
    ys = np.random.randint(0, h, num_drops)
    lengths = np.random.randint(10, 20, num_drops)
    
    for x, y, l in zip(xs, ys, lengths):
        cv2.line(result, (x, y), (x + 2, y + l), (200, 200, 200), 1)
    
    return result

def add_fog(image):
    try:
        transform = A.RandomFog(fog_coef_lower=0.5, fog_coef_upper=0.7, alpha_coef=0.1, always_apply=True)
        return transform(image=image)['image']
    except Exception:
        # Custom fallback
        row, col, ch = image.shape
        fog = np.ones((row, col, 3), dtype=np.uint8) * 200
        return cv2.addWeighted(image, 0.5, fog, 0.5, 0)

def add_snow(image):
    try:
        transform = A.RandomSnow(brightness_coeff=2.5, snow_point_lower=0.2, snow_point_upper=0.4, always_apply=True)
        return transform(image=image)['image']
    except Exception:
        result = image.copy()
        h, w = result.shape[:2]
        num_flakes = int(0.01 * h * w)
        xs = np.random.randint(0, w, num_flakes)
        ys = np.random.randint(0, h, num_flakes)
        
        for x, y in zip(xs, ys):
            cv2.circle(result, (x, y), 2, (255, 255, 255), -1)
        return result

def add_wind_blur(image):
    try:
        transform = A.MotionBlur(blur_limit=(15, 25), always_apply=True)
        return transform(image=image)['image']
    except Exception:
        kernel = np.zeros((15, 15))
        kernel[int((15-1)/2), :] = np.ones(15)
        kernel = kernel / 15
        return cv2.filter2D(image, -1, kernel)

def simulate_weather(input_dir='dataset/images/test', output_base='dataset/images/test_weather'):
    print("\n--- Starting Weather Simulation ---")
    if not os.path.exists(input_dir):
        print(f"Error: {input_dir} not found.")
        return

    conditions = ['rain', 'fog', 'snow', 'wind']
    funcs = [add_rain, add_fog, add_snow, add_wind_blur]
    
    images = glob.glob(os.path.join(input_dir, '*.jpg')) + glob.glob(os.path.join(input_dir, '*.png'))
    if not images:
         print(f"No images found in {input_dir}")
         return

    for cond, func in zip(conditions, funcs):
        out_dir = os.path.join(output_base, cond)
        os.makedirs(out_dir, exist_ok=True)
        
        lbl_out_dir = out_dir.replace('images', 'labels')
        os.makedirs(lbl_out_dir, exist_ok=True)
        
        print(f"Simulating {cond.upper()} condition on {len(images)} images...")
        
        for img_path in tqdm(images):
            basename = os.path.basename(img_path)
            img = cv2.imread(img_path)
            if img is not None:
                degraded = func(img)
                cv2.imwrite(os.path.join(out_dir, basename), degraded)
                
                # Copy label file
                lbl_name = os.path.splitext(basename)[0] + '.txt'
                lbl_path = os.path.join(input_dir.replace('images', 'labels'), lbl_name)
                if os.path.exists(lbl_path):
                    import shutil
                    shutil.copy(lbl_path, os.path.join(lbl_out_dir, lbl_name))
                
    print(f"Weather simulation completed. Output saved to {output_base}")

if __name__ == "__main__":
    simulate_weather()

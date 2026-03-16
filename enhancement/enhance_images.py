import cv2
import numpy as np
import os
import glob
from tqdm import tqdm

def apply_hist_eq(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def apply_claHE(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def apply_gamma(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def apply_dehaze(image, tmin=0.1, w=0.95):
    # Simplified Dark Channel Prior-like dehazing approximation
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dark = cv2.erode(gray, kernel)
    
    A = max(dark.max(), 1) # Atmosphere light
    transmission = 1 - w * (dark / A)
    transmission = np.clip(transmission, tmin, 1)
    
    result = np.zeros_like(image, dtype=np.float32)
    for i in range(3):
        result[:,:,i] = (image[:,:,i] - A) / transmission + A
        
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_sharpen(image):
    kernel = np.array([[0, -1, 0], 
                       [-1, 5,-1], 
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def enhance_images(input_base='dataset/images/test_weather', output_base='dataset/images/test_enhanced'):
    print("\n--- Starting Image Enhancement ---")
    
    conditions = ['rain', 'fog', 'snow', 'wind']
    techniques = {
        'hist_eq': apply_hist_eq,
        'clahe': apply_claHE,
        'gamma': apply_gamma,
        'dehaze': apply_dehaze,
        'sharpen': apply_sharpen
    }
    
    for cond in conditions:
        in_dir = os.path.join(input_base, cond)
        if not os.path.exists(in_dir):
            continue
            
        images = glob.glob(os.path.join(in_dir, '*.png')) + glob.glob(os.path.join(in_dir, '*.jpg'))
        
        for tech_name, func in techniques.items():
            out_dir = os.path.join(output_base, cond, tech_name)
            os.makedirs(out_dir, exist_ok=True)
            
            lbl_out_dir = out_dir.replace('images', 'labels')
            os.makedirs(lbl_out_dir, exist_ok=True)
            
            print(f"Applying {tech_name} to {cond} images ({len(images)} files)...")
            
            for img_path in tqdm(images, leave=False):
                basename = os.path.basename(img_path)
                img = cv2.imread(img_path)
                if img is not None:
                    enhanced = func(img)
                    cv2.imwrite(os.path.join(out_dir, basename), enhanced)
                    
                    # Copy label file
                    lbl_name = os.path.splitext(basename)[0] + '.txt'
                    lbl_path = os.path.join(in_dir.replace('images', 'labels'), lbl_name)
                    if os.path.exists(lbl_path):
                        import shutil
                        shutil.copy(lbl_path, os.path.join(lbl_out_dir, lbl_name))
                    
    print(f"Enhancement completed. Output saved to {output_base}")

if __name__ == "__main__":
    enhance_images()

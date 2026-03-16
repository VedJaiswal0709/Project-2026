import cv2
import numpy as np
import os
import glob
import logging
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------------------------------------
# 1. Enhancement Techniques
# ---------------------------------------------------------

def apply_histogram_equalization(image):
    """
    Applies standard Histogram Equalization to the intensity (Y) channel.
    """
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    channels = list(cv2.split(ycrcb))
    channels[0] = cv2.equalizeHist(channels[0])
    merged = cv2.merge(channels)
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8,8)):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def apply_gamma_correction(image, gamma=1.5):
    """
    Applies Gamma Correction to brighten or darken an image.
    (Gamma > 1 darkens, Gamma < 1 brightens).
    Using gamma=1.5 as default for overly bright fog/snow.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def apply_dark_channel_dehaze(image, w=0.95, t0=0.1):
    """
    A simplified Dark Channel Prior (DCP) approach for dehazing.
    Note: Full DCP requires guided filtering, using basic approximation here.
    """
    # 1. Get Dark Channel
    b, g, r = cv2.split(image)
    min_img = cv2.min(cv2.min(b, g), r)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dark_channel = cv2.erode(min_img, kernel)

    # 2. Get Atmospheric Light (A)
    flat_dark = dark_channel.flatten()
    flat_img = image.reshape((-1, 3))
    num_px = int(max(math.floor(len(flat_dark) / 1000), 1))
    indices = np.argpartition(flat_dark, -num_px)[-num_px:]
    A = np.max(flat_img[indices], axis=0)

    # 3. Get Transmission (t)
    norm_img = image.astype(np.float64) / A
    norm_min = cv2.min(cv2.min(norm_img[:, :, 0], norm_img[:, :, 1]), norm_img[:, :, 2])
    dark_norm = cv2.erode(norm_min, kernel)
    t = 1 - w * dark_norm

    # 4. Recover Scene Radiance
    t = np.maximum(t, t0)
    t_mat = np.zeros_like(image, dtype=np.float64)
    for i in range(3):
        t_mat[:, :, i] = t
        
    recovered = ((image.astype(np.float64) - A) / t_mat) + A
    return np.clip(recovered, 0, 255).astype(np.uint8)

def apply_sharpening_filter(image):
    """
    Applies a standard Laplacian-based sharpening spatial filter.
    Useful for wind/motion blur recovery.
    """
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)


# ---------------------------------------------------------
# 2. Metric Computation
# ---------------------------------------------------------

def calculate_brightness(image):
    """Returns average luminance of the image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:, :, 2])

def calculate_contrast(image):
    """Returns RMS contrast (standard deviation of pixel intensities)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.std()

def calculate_sharpness(image):
    """Returns variance of the Laplacian (higher means sharper edges)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def compute_improvements(original, enhanced):
    """
    Computes the percentage improvement of Contrast, Brightness, and Sharpness.
    """
    b_orig, b_enh = calculate_brightness(original), calculate_brightness(enhanced)
    c_orig, c_enh = calculate_contrast(original), calculate_contrast(enhanced)
    s_orig, s_enh = calculate_sharpness(original), calculate_sharpness(enhanced)
    
    # Avoid division by zero
    diff_b = ((b_enh - b_orig) / (b_orig + 1e-6)) * 100
    diff_c = ((c_enh - c_orig) / (c_orig + 1e-6)) * 100
    diff_s = ((s_enh - s_orig) / (s_orig + 1e-6)) * 100
    
    return diff_b, diff_c, diff_s


# ---------------------------------------------------------
# 3. Main Processing Pipeline
# ---------------------------------------------------------

def process_enhancements(input_dir, output_dir):
    """
    Reads degraded weather images, applies all 5 enhancement techniques,
    computes metric improvements, and saves them to `enhancement_results/`.
    """
    if not os.path.exists(input_dir):
        logging.error(f"Input directory not found: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Define the mapping of technique names to functions
    techniques = {
        'hist_eq': apply_histogram_equalization,
        'clahe': apply_clahe,
        'gamma': apply_gamma_correction,
        'dehaze': apply_dark_channel_dehaze,
        'sharpen': apply_sharpening_filter
    }
    
    # Create subfolders for each technique
    for name in techniques.keys():
        os.makedirs(os.path.join(output_dir, name), exist_ok=True)

    image_paths = glob.glob(os.path.join(input_dir, '*.*'))
    valid_images = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not valid_images:
        logging.warning(f"No valid images found in {input_dir}")
        return

    logging.info(f"Starting enhancement processing for {len(valid_images)} images...")

    for img_path in valid_images:
        filename = os.path.basename(img_path)
        original_img = cv2.imread(img_path)
        
        if original_img is None:
            continue
            
        logging.info(f"--- Processing {filename} ---")
            
        for name, func in techniques.items():
            try:
                # Apply enhancement
                enhanced_img = func(original_img)
                
                # Compute improvements
                diff_b, diff_c, diff_s = compute_improvements(original_img, enhanced_img)
                
                # Log metrics
                logging.info(f"  [{name.upper()}] Metrics Delta -> Brightness: {diff_b:+.2f}% | Contrast: {diff_c:+.2f}% | Sharpness: {diff_s:+.2f}%")
                
                # Save result
                save_path = os.path.join(output_dir, name, filename)
                cv2.imwrite(save_path, enhanced_img)
                
            except Exception as e:
                logging.error(f"Error applying {name} to {filename}: {e}")

    logging.info(f"Enhancement pipeline complete. Results saved in: {output_dir}")

if __name__ == "__main__":
    # --- Configuration ---
    # Point this to a folder with weather-degraded images (e.g. from previous steps)
    DEGRADED_IMAGES_DIR = "dataset/weather_simulations/fog_images" 
    
    # Master directory where all enhanced images will be routed
    ENHANCEMENT_OUTPUT_DIR = "enhancement_results"
    
    # Uncomment to execute:
    # process_enhancements(DEGRADED_IMAGES_DIR, ENHANCEMENT_OUTPUT_DIR)
    
    logging.info("Enhancement script ready. Configure 'DEGRADED_IMAGES_DIR' in main to run.")
    pass

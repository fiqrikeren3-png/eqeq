import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from collections import defaultdict

# --- KONFIGURASI ---
DISTANCES = [1, 2, 3]
ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]
# Perhatikan baris di bawah ini harus lengkap ada kurung tutup ']'
PROPS = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

LEVELS = 32
IMG_SIZE = (200, 200)

def extract_glcm_features(gray):
    """
    Input: grayscale uint8 image
    Output: vektor fitur 1D numpy array
    """
    # Kuantisasi warna jadi 32 level
    gray_quantized = (gray // (256 // LEVELS)).astype(np.uint8)

    # Hitung GLCM
    glcm = graycomatrix(
        gray_quantized,
        distances=DISTANCES,
        angles=ANGLES,
        levels=LEVELS,
        symmetric=True,
        normed=True
    )

    features = []

    for prop in PROPS:
        values = graycoprops(glcm, prop)
        # Ambil rata-rata dan range dari semua sudut
        mean_val = np.mean(values)
        range_val = np.ptp(values)
        features.append(mean_val)
        features.append(range_val)

    return np.array(features)

def analyze_color(img_path):
    img = cv2.imread(img_path)
    if img is None: return [0,0,0]
    img = cv2.resize(img, IMG_SIZE)
    mean_r = np.mean(img[:, :, 2])
    mean_g = np.mean(img[:, :, 1])
    mean_b = np.mean(img[:, :, 0])
    return [mean_r, mean_g, mean_b]

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Unable to load image from {img_path}")
    img = cv2.resize(img, IMG_SIZE)

    # --- FITUR WARNA ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    std_s = np.std(hsv[:, :, 1])
    mean_v = np.mean(hsv[:, :, 2])
    std_v = np.std(hsv[:, :, 2])

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    mean_l = np.mean(lab[:, :, 0])
    std_l = np.std(lab[:, :, 0])

    color_features = np.array([std_s, mean_v, std_v, mean_l, std_l])

    # --- FITUR TEKSTUR ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    entropy_val = shannon_entropy(gray)
    glcm_features = extract_glcm_features(gray)

    combined_features = np.concatenate(([entropy_val], color_features, glcm_features))
    return combined_features
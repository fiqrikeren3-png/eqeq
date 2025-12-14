import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from collections import defaultdict

# Konfigurasi GLCM
DISTANCES = [1, 2, 4, 8]
ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
PROPS = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
LEVELS = 256
IMG_SIZE = (200, 200)

def extract_glcm_features(gray):
    """
    Input: grayscale uint8 image (200x200)
    Output: vektor fitur 1D numpy array
    """
    glcm = graycomatrix(
        gray,
        distances=DISTANCES,
        angles=ANGLES,
        levels=LEVELS,
        symmetric=True,
        normed=True
    )
    features = np.array([])
    for prop in PROPS:
        values = graycoprops(glcm, prop)
        features = np.concatenate((features, values.flatten()))
    return features.flatten()

def analyze_color(img_path):
    """
    Analisis warna menggunakan mean RGB values
    Output: list of mean R, G, B values
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Unable to load image from {img_path}")
    img = cv2.resize(img, IMG_SIZE)

    # Hitung mean R, G, B
    mean_r = np.mean(img[:, :, 2])  # OpenCV is BGR
    mean_g = np.mean(img[:, :, 1])
    mean_b = np.mean(img[:, :, 0])

    return [mean_r, mean_g, mean_b]

def preprocess_image(img_path):
    """
    Load -> resize -> ekstraksi fitur GLCM + warna (mean, std R, G, B + LAB)
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Unable to load image from {img_path}")
    img = cv2.resize(img, IMG_SIZE)

    # Ekstraksi fitur warna RGB
    mean_r = np.mean(img[:, :, 2])  # OpenCV is BGR
    mean_g = np.mean(img[:, :, 1])
    mean_b = np.mean(img[:, :, 0])
    std_r = np.std(img[:, :, 2])
    std_g = np.std(img[:, :, 1])
    std_b = np.std(img[:, :, 0])

    # Konversi ke LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    mean_l = np.mean(lab[:, :, 0])
    mean_a = np.mean(lab[:, :, 1])
    mean_bb = np.mean(lab[:, :, 2])  # bb to avoid conflict with b
    std_l = np.std(lab[:, :, 0])
    std_a = np.std(lab[:, :, 1])
    std_bb = np.std(lab[:, :, 2])

    # Konversi ke HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean_h = np.mean(hsv[:, :, 0])
    mean_s = np.mean(hsv[:, :, 1])
    mean_v = np.mean(hsv[:, :, 2])
    std_h = np.std(hsv[:, :, 0])
    std_s = np.std(hsv[:, :, 1])
    std_v = np.std(hsv[:, :, 2])

    color_features = [mean_r, mean_g, mean_b, std_r, std_g, std_b, mean_l, mean_a, mean_bb, std_l, std_a, std_bb, mean_h, mean_s, mean_v, std_h, std_s, std_v]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    glcm_features = extract_glcm_features(gray)

    # Gabungkan fitur GLCM dan warna
    combined_features = np.concatenate((glcm_features, color_features))
    return combined_features

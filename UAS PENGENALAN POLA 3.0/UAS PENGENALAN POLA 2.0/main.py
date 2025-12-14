import os
import glob
import cv2
import numpy as np
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import graycomatrix, graycoprops
import joblib
import argparse

def extract_glcm_features(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """
    Ekstrak fitur GLCM dari gambar.
    """
    # Resize gambar ke ukuran standar
    image = cv2.resize(image, (128, 128))
    # Konversi ke grayscale jika belum
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Hitung GLCM
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

    # Ekstrak fitur Haralick
    features = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
        features.append(graycoprops(glcm, prop).mean())

    return np.array(features)

def load_dataset(dataset_path):
    """
    Muat dataset gambar dari folder.
    """
    features = []
    labels = []

    if not os.path.exists(dataset_path):
        print(f"Folder dataset '{dataset_path}' tidak ditemukan. Silakan buat folder dan tambahkan gambar.")
        return np.array([]), np.array([])

    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    if not classes:
        print("Tidak ada subfolder kelas di folder dataset. Silakan buat subfolder seperti 'plastik', 'kayu', 'besi'.")
        return np.array([]), np.array([])

    print(f"Ditemukan {len(classes)} kelas: {classes}")

    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        image_paths = glob.glob(os.path.join(class_path, '*.jpg')) + \
                     glob.glob(os.path.join(class_path, '*.png')) + \
                     glob.glob(os.path.join(class_path, '*.jpeg'))

        print(f"Memuat {len(image_paths)} gambar dari kelas '{class_name}'")

        for image_path in image_paths:
            image = cv2.imread(image_path)
            if image is not None:
                feature = extract_glcm_features(image)
                features.append(feature)
                labels.append(class_name)

    return np.array(features), np.array(labels)

def train_model(dataset_path, model_path='texture_classifier.pkl'):
    """
    Latih model SVM pada dataset.
    """
    print("Memuat dataset...")
    features, labels = load_dataset(dataset_path)

    if len(features) == 0:
        print("Dataset kosong. Tidak dapat melatih model.")
        return None, None

    print(f"Dataset dimuat: {len(features)} sampel, {len(np.unique(labels))} kelas")

    # Encode label
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)

    # Latih SVM
    clf = svm.SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)

    # Evaluasi
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Akurasi: {accuracy:.2f}")

    # Simpan model
    joblib.dump((clf, le), model_path)
    print(f"Model disimpan ke '{model_path}'")

    return clf, le

def classify_image(image_path, model_path='texture_classifier.pkl'):
    """
    Klasifikasikan gambar menggunakan model terlatih.
    """
    if not os.path.exists(model_path):
        print(f"Model '{model_path}' tidak ditemukan. Jalankan training terlebih dahulu.")
        return None

    # Muat model
    clf, le = joblib.load(model_path)

    # Muat dan proses gambar
    image = cv2.imread(image_path)
    if image is None:
        print(f"Tidak dapat memuat gambar '{image_path}'")
        return None

    feature = extract_glcm_features(image)
    feature = feature.reshape(1, -1)

    # Prediksi
    pred_encoded = clf.predict(feature)[0]
    pred_class = le.inverse_transform([pred_encoded])[0]
    prob = clf.predict_proba(feature)[0][pred_encoded]

    print(f"Gambar '{image_path}' diklasifikasikan sebagai: {pred_class} (probabilitas: {prob:.2f})")

    return pred_class

def main():
    parser = argparse.ArgumentParser(description='Klasifikasi Tekstur Permukaan')
    parser.add_argument('--train', action='store_true', help='Latih model pada dataset')
    parser.add_argument('--classify', type=str, help='Klasifikasikan gambar (path ke gambar)')
    parser.add_argument('--dataset', type=str, default='dataset', help='Path ke folder dataset (default: dataset)')

    args = parser.parse_args()

    if args.train:
        train_model(args.dataset)
    elif args.classify:
        classify_image(args.classify)
    else:
        print("Gunakan --train untuk melatih model atau --classify untuk mengklasifikasikan gambar.")
        print("Contoh:")
        print("  python main.py --train")
        print("  python main.py --classify path/to/image.jpg")

if __name__ == "__main__":
    main()

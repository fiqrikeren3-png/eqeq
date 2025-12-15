import numpy as np
import cv2
from preprocessing_glcm import extract_glcm_features, DISTANCES, ANGLES, PROPS
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "dataset", "train")
MODEL_PATH = "glcm_svm_model.joblib"
IMG_SIZE = (200, 200)
K_NEIGH = 5

def load_dataset(folder):
    X, y = [], []
    classes = sorted(os.listdir(folder))
    for cls in classes:
        cls_path = os.path.join(folder, cls)
        if not os.path.isdir(cls_path): continue
        for fname in os.listdir(cls_path):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')): continue
            img_path = os.path.join(cls_path, fname)
            img = cv2.imread(img_path)
            if img is None: continue
            feats = None
            try:
                feats = extract_glcm_features(cv2.cvtColor(cv2.resize(img, IMG_SIZE), cv2.COLOR_BGR2GRAY))
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
            X.append(feats)
            y.append(cls)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    X, y = load_dataset(DATA_DIR)
    if len(X) == 0:
        print("Dataset kosong atau gambar tidak valid.")
        exit()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Encode labels to integers
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Hyperparameter tuning with GridSearchCV for SVM
    param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}
    grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train_encoded)
    best_model = grid_search.best_estimator_
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)

    y_pred_encoded = best_model.predict(X_test_scaled)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    model_data = {
        'model': best_model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'classes': np.unique(y),
        'img_size': IMG_SIZE,
        'distances': DISTANCES,
        'angles': ANGLES,
        'props': PROPS
    }
    joblib.dump(model_data, MODEL_PATH)
    print("Model saved to", MODEL_PATH)

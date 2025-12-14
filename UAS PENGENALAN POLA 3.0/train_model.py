import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from preprocessing_glcm import preprocess_image, DISTANCES, ANGLES, PROPS, IMG_SIZE

DATA_DIR = "dataset/train"
MODEL_PATH = "glcm_svm_model.joblib"

def load_dataset():
    X = []
    y = []
    classes = sorted(os.listdir(DATA_DIR))
    for cls in classes:
        class_path = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(class_path):
            continue
        for file in os.listdir(class_path):
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(class_path, file)
                try:
                    features = preprocess_image(img_path)
                    X.append(features)
                    y.append(cls)
                except:
                    print("Error membaca:", img_path)
    return np.array(X), np.array(y), classes

def train_improved_model():
    print("[1] Loading dataset...")
    X, y, classes = load_dataset()
    print("Total data :", len(X))
    print("Kelas      :", classes)
    print("Feature dim:", X.shape)

    # Split dataset untuk evaluasi
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Preprocessing
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Hyperparameter tuning untuk SVM
    param_grid_svm = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'kernel': ['rbf', 'linear']
    }

    print("[2] Tuning hyperparameters for SVM...")
    grid_search_svm = GridSearchCV(
        SVC(probability=True, random_state=42),
        param_grid_svm,
        cv=3,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search_svm.fit(X_train_scaled, y_train_encoded)

    best_svm = grid_search_svm.best_estimator_
    print(f"SVM Best parameters: {grid_search_svm.best_params_}")
    print(".3f")

    # Hyperparameter tuning untuk RandomForest
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }

    print("[3] Tuning hyperparameters for RandomForest...")
    grid_search_rf = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid_rf,
        cv=3,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search_rf.fit(X_train_scaled, y_train_encoded)

    best_rf = grid_search_rf.best_estimator_
    print(f"RF Best parameters: {grid_search_rf.best_params_}")
    print(".3f")

    # Evaluasi pada test set untuk kedua model
    y_pred_svm = best_svm.predict(X_test_scaled)
    y_pred_rf = best_rf.predict(X_test_scaled)

    print("\n[4] SVM Classification Report:")
    print(classification_report(y_test_encoded, y_pred_svm, target_names=classes))

    print("\n[5] RF Classification Report:")
    print(classification_report(y_test_encoded, y_pred_rf, target_names=classes))

    # Cross-validation score untuk kedua model
    cv_scores_svm = cross_val_score(best_svm, X_train_scaled, y_train_encoded, cv=5)
    cv_scores_rf = cross_val_score(best_rf, X_train_scaled, y_train_encoded, cv=5)
    print(".3f")
    print(".3f")

    # Bandingkan akurasi test set
    acc_svm = accuracy_score(y_test_encoded, y_pred_svm)
    acc_rf = accuracy_score(y_test_encoded, y_pred_rf)
    print(".3f")
    print(".3f")

    # Pilih model terbaik berdasarkan akurasi test set
    if acc_svm >= acc_rf:
        best_model = best_svm
        best_params = grid_search_svm.best_params_
        cv_score = cv_scores_svm.mean()
        model_name = "SVM"
    else:
        best_model = best_rf
        best_params = grid_search_rf.best_params_
        cv_score = cv_scores_rf.mean()
        model_name = "RandomForest"

    print(f"\n[6] Best model selected: {model_name}")

    # Simpan model terbaik
    model_pack = {
        "model": best_model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "classes": classes,
        "distances": DISTANCES,
        "angles": ANGLES,
        "props": PROPS,
        "img_size": IMG_SIZE,
        "best_params": best_params,
        "cv_score": cv_score,
        "model_name": model_name
    }
    joblib.dump(model_pack, MODEL_PATH)
    print("\n[7] Model saved as:", MODEL_PATH)
    return best_model, scaler, label_encoder, classes

if __name__ == "__main__":
    train_improved_model()

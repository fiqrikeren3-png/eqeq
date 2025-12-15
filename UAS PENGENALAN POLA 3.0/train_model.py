import os
import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from preprocessing_glcm import preprocess_image, DISTANCES, ANGLES, PROPS, IMG_SIZE

DATA_DIR = os.path.join(os.path.dirname(__file__), "dataset", "train")
MODEL_PATH = "glcm_best_model.joblib"

def load_dataset():
    X = []
    y = []
    classes = sorted(os.listdir(DATA_DIR))
    print("Membaca dataset...")
    for cls in classes:
        class_path = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(class_path):
            continue
        files = os.listdir(class_path)
        print(f" - Kelas {cls}: {len(files)} gambar")
        for file in files:
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(class_path, file)
                try:
                    features = preprocess_image(img_path)
                    X.append(features)
                    y.append(cls)
                except Exception as e:
                    print(f"Error membaca {file}: {e}")
    return np.array(X), np.array(y), classes

def train_optimized_model():
    # 1. Load Data
    X, y, classes = load_dataset()
    if len(X) == 0:
        print("Data tidak ditemukan! Jalankan augment_data.py dulu atau cek folder dataset.")
        return

    # 2. Split Data (80% Train, 20% Test) - Lebih banyak untuk training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 3. Preprocessing
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # 4. Definisi Model dan Parameter untuk Tuning
    models_params = {
        'SVM': {
            'model': SVC(probability=True, random_state=42),
            'params': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.1, 0.01],
                'kernel': ['rbf', 'linear', 'poly']
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5]
            }
        }
    }

    best_score = 0
    best_model_info = None
    best_clf = None

    print("\n=== Memulai Pencarian Model Terbaik (Grid Search) ===")
    
    for model_name, mp in models_params.items():
        print(f"\nTraining {model_name}...")
        clf = GridSearchCV(mp['model'], mp['params'], cv=5, n_jobs=-1, scoring='accuracy')
        clf.fit(X_train_scaled, y_train_encoded)
        
        # Evaluasi
        score = clf.best_score_
        print(f" -> Best CV Score: {score:.4f}")
        print(f" -> Best Params: {clf.best_params_}")
        
        # Cek performa di data test
        y_pred = clf.best_estimator_.predict(X_test_scaled)
        test_acc = accuracy_score(y_test_encoded, y_pred)
        print(f" -> Test Accuracy: {test_acc:.4f}")

        if test_acc > best_score:
            best_score = test_acc
            best_clf = clf.best_estimator_
            best_model_info = {
                "name": model_name,
                "params": clf.best_params_,
                "accuracy": test_acc
            }

    print("\n=============================================")
    print(f"MODEL TERBAIK: {best_model_info['name']}")
    print(f"Akurasi: {best_model_info['accuracy']:.4f}")
    print("=============================================")

    # 5. Simpan Model Terbaik
    model_pack = {
        "model": best_clf,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "classes": classes,
        "distances": DISTANCES,
        "angles": ANGLES,
        "props": PROPS,
        "img_size": IMG_SIZE,
        "model_info": best_model_info
    }
    joblib.dump(model_pack, MODEL_PATH)
    print(f"Model tersimpan di: {MODEL_PATH}")

    # Tampilkan Report Akhir
    y_final_pred = best_clf.predict(X_test_scaled)
    print("\nClassification Report (Best Model):")
    print(classification_report(y_test_encoded, y_final_pred, target_names=classes))

if __name__ == "__main__":
    train_optimized_model()
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

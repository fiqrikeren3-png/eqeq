import cv2
import numpy as np
import joblib
from skimage.feature import graycomatrix, graycoprops

MODEL_PATH = "glcm_svm_model.joblib"
ROI_SIZE = (200, 200)

def extract_glcm_features(gray, distances, angles, levels, props):
    glcm = graycomatrix(gray, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    feats = []
    for p in props:
        arr = graycoprops(glcm, p)
        feats.extend(arr.flatten().tolist())
    return np.array(feats).reshape(1, -1)

if __name__ == "__main__":
    data = joblib.load(MODEL_PATH)
    model = data['model']
    scaler = data.get('scaler')
    label_encoder = data['label_encoder']
    distances = data['distances']
    angles = data['angles']
    props = data['props']
    levels = 256
    img_size = tuple(data.get('img_size', ROI_SIZE))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret: break
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        rw, rh = img_size
        x1, y1 = cx - rw//2, cy - rh//2
        x2, y2 = x1 + rw, y1 + rh
        roi = frame[y1:y2, x1:x2]
        if roi.shape[0] != rh or roi.shape[1] != rw:
            roi = cv2.resize(roi, img_size)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        feats = extract_glcm_features(gray, distances, angles, levels, props)
        if scaler:
            feats = scaler.transform(feats)
        pred_encoded = model.predict(feats)[0]
        label = label_encoder.inverse_transform([pred_encoded])[0]
        conf_text = ""
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(feats)[0]
            conf = probs.max()
            conf_text = f"{conf*100:.2f}%"
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        # Tampilkan label dengan font besar, warna kuning, outline hitam
        text = f"Pred: {label} {conf_text}"
        # Outline hitam
        cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 5)
        # Teks utama kuning
        cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,255), 3)
        cv2.imshow("Realtime Texture Classification", frame)
        cv2.imshow("ROI", roi)
        k = cv2.waitKey(1) & 0xFF
        # Tekan ESC (27) atau 'q' (ord('q')==113) untuk keluar
        if k == 27 or k == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()

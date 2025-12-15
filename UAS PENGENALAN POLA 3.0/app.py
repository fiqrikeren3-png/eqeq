
from flask import Flask, request, render_template_string
import joblib
import cv2
import numpy as np
from preprocessing_glcm import preprocess_image

app = Flask(__name__)
MODEL_PATH = "glcm_best_model.joblib"

HTML = '''
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Klasifikasi Tekstur Permukaan</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Roboto', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
        }
        .container {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            max-width: 600px;
            width: 100%;
            text-align: center;
        }
        h2 {
            color: #4a4a4a;
            margin-bottom: 20px;
        }
        form {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 10px;
            font-weight: 500;
        }
        input[type="file"] {
            margin-bottom: 15px;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 5px;
            width: 100%;
            box-sizing: border-box;
        }
        button {
            background: #667eea;
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            transition: background 0.3s;
        }
        button:hover {
            background: #5a6fd8;
        }
        hr {
            border: none;
            height: 1px;
            background: #ddd;
            margin: 20px 0;
        }
        h4 {
            color: #4a4a4a;
            margin-bottom: 10px;
        }
        img {
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            margin-bottom: 15px;
        }
        h3 {
            color: #333;
        }
        p {
            font-size: 18px;
            margin-bottom: 15px;
        }
        ul {
            list-style: none;
            padding: 0;
        }
        li {
            background: #f9f9f9;
            margin: 8px 0;
            padding: 12px;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .progress-bar {
            width: 150px;
            height: 8px;
            background: #e0e0e0;
            border-radius: 4px;
            overflow: hidden;
            margin-left: 10px;
        }
        .progress-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }
        .progress-fill.high { background: #4CAF50; }
        .progress-fill.medium { background: #FF9800; }
        .progress-fill.low { background: #f44336; }
        .class-name {
            font-weight: 500;
        }
        .percentage {
            font-weight: bold;
            font-size: 14px;
        }
        @media (max-width: 768px) {
            .container {
                padding: 20px;
            }
            button {
                width: 100%;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Klasifikasi Tekstur (Kayu, Kain, Metal)</h2>
        <form method="POST" enctype="multipart/form-data">
            <label>Masukkan Gambar Tekstur:</label>
            <input type="file" name="file" required>
            <button type="submit">Klasifikasi</button>
        </form>
        {% if img_data %}
            <hr>
            <h4>Gambar yang diupload:</h4>
            <img src="data:image/jpeg;base64,{{ img_data }}" alt="Uploaded Image" />
        {% endif %}
        {% if prediction %}
            <hr>
            <h3>Hasil Prediksi:</h3>
            <p><b>{{ prediction }}</b></p>
            {% if probs %}
                <h4>Persentase Kelas:</h4>
                <ul>
                {% for k, v in probs.items() %}
                    <li>
                        <span class="class-name">{{ k }}</span>
                        <div style="display: flex; align-items: center;">
                            <span class="percentage">{{ v }}</span>
                            <div class="progress-bar">
                                <div class="progress-fill {% if probs_raw[k] > 70 %}high{% elif probs_raw[k] > 30 %}medium{% else %}low{% endif %}" style="width: {{ probs_raw[k] }}%;"></div>
                            </div>
                        </div>
                    </li>
                {% endfor %}
                </ul>
                {% if above_50 %}
                    <p style="color: green; font-weight: bold;">Ada kelas dengan persentase di atas 50%!</p>
                {% else %}
                    <p style="color: red;">Tidak ada kelas dengan persentase di atas 50%.</p>
                {% endif %}
            {% endif %}

        {% endif %}
    </div>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probs = None
    probs_raw = None
    img_data = None
    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        filepath = 'temp_upload.jpg'
        file.save(filepath)
        try:
            model_pack = joblib.load(MODEL_PATH)
            model = model_pack["model"]
            scaler = model_pack.get("scaler")
            label_encoder = model_pack.get("label_encoder")
            classes = model_pack["classes"]
            features = preprocess_image(filepath)
            features = np.array([features])
            if scaler:
                features = scaler.transform(features)
            # Ambil probabilitas semua kelas terlebih dahulu
            if hasattr(model, "predict_proba"):
                prob_array = model.predict_proba(features)[0]
                pred_encoded = np.argmax(prob_array)  # Pastikan prediksi sesuai dengan probabilitas tertinggi
                if label_encoder:
                    prediction = label_encoder.inverse_transform([pred_encoded])[0]
                    probs = {label_encoder.inverse_transform([i])[0]: f"{prob_array[i]*100:.2f}%" for i in range(len(classes))}
                    probs_raw = {label_encoder.inverse_transform([i])[0]: prob_array[i]*100 for i in range(len(classes))}
                else:
                    prediction = classes[pred_encoded]
                    probs = {classes[i]: f"{prob_array[i]*100:.2f}%" for i in range(len(classes))}
                    probs_raw = {classes[i]: prob_array[i]*100 for i in range(len(classes))}
                # Deteksi jika ada persentase di atas 50%
                above_50 = any(prob > 50 for prob in probs_raw.values())
            else:
                # Fallback jika model tidak support predict_proba
                pred_encoded = model.predict(features)[0]
                if label_encoder:
                    prediction = label_encoder.inverse_transform([pred_encoded])[0]
                else:
                    prediction = classes[pred_encoded]
                above_50 = False
                probs_raw = None
        except Exception as e:
            prediction = f"Error processing image: {str(e)}"
            probs = None
        # Encode gambar untuk preview
        import base64
        with open(filepath, "rb") as img_f:
            img_data = base64.b64encode(img_f.read()).decode("utf-8")
    return render_template_string(HTML, prediction=prediction, probs=probs, probs_raw=probs_raw, img_data=img_data, above_50=above_50 if 'above_50' in locals() else False)

if __name__ == '__main__':
    app.run(debug=True)

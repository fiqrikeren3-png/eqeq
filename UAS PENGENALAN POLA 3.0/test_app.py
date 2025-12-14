import requests
import os

# Test the Flask app with sample images
url = 'http://127.0.0.1:5000'

# Sample images from dataset
test_images = [
    'dataset/train/kain/kain_dummy.jpg',
    'dataset/train/kayu/kayu_dummy.jpg',
    'dataset/train/metal/metal_dummy.jpg'
]

for img_path in test_images:
    if os.path.exists(img_path):
        with open(img_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files)
            print(f"Testing {img_path}:")
            if response.status_code == 200:
                # Parse the HTML to extract prediction
                html = response.text
                # Simple parsing for prediction
                if 'Hasil Prediksi:' in html:
                    start = html.find('<p><b>') + 6
                    end = html.find('</b></p>', start)
                    prediction = html[start:end]
                    print(f"Prediction: {prediction}")
                else:
                    print("No prediction found")
            else:
                print(f"Error: {response.status_code}")
            print("-" * 50)
    else:
        print(f"Image {img_path} not found")

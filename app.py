# app.py  —  Cat & Dog Classifier  |  Flask Web App
# ─────────────────────────────────────────────────
 
import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
 
# ── 1. Initialise Flask ───────────────────────────
app = Flask(__name__)
 
# ── 2. Load the Trained Model Once at Startup ────
# Loading inside a function would reload it on every
# request, which would be extremely slow.
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.h5')
model = load_model(MODEL_PATH)
print('✅  model.h5 loaded successfully')
 
# ── 3. Class Labels ───────────────────────────────
# image_dataset_from_directory sorts folders alphabetically.
# 'cat' comes before 'dog', so:  cat = 0,  dog = 1
# Your sigmoid output: >= 0.5 → Dog,  < 0.5 → Cat
LABELS = {0: 'Cat 🐱', 1: 'Dog 🐶'}
 
# ── 4. Home Route — Serve the HTML Page ──────────
@app.route('/')
def home():
    return render_template('index.html')
 
# ── 5. Prediction Route ───────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    # Check that a file was actually uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
 
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
 
    try:
        # Read the uploaded image bytes
        img_bytes = file.read()
        img_pil   = Image.open(io.BytesIO(img_bytes)).convert('RGB')
 
        # Resize to 200x200 — MUST match your training image_size=(200,200)
        img_pil   = img_pil.resize((200, 200))
 
        # Convert to numpy array and add batch dimension
        img_array = image.img_to_array(img_pil)        # shape: (200, 200, 3)
        img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 200, 200, 3)
 
        # Run prediction
        raw_output   = float(model.predict(img_array)[0][0])
 
        # Map to label and confidence
        if raw_output >= 0.5:
            label      = LABELS[1]
            confidence = round(raw_output * 100, 1)
        else:
            label      = LABELS[0]
            confidence = round((1 - raw_output) * 100, 1)
 
        return jsonify({
            'prediction': label,
            'confidence': f'{confidence}%',
            'raw_score':  round(raw_output, 4)
        })
 
    except Exception as e:
        return jsonify({'error': str(e)}), 500
 
# ── 6. Entry Point ────────────────────────────────
if __name__ == '__main__':
    # Render injects the PORT environment variable.
    # Locally this defaults to 5000.
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

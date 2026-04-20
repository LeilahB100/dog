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

# Limit uploads to 10 MB — prevents RAM exhaustion on Render's free tier
# that would cause "Unexpected end of JSON input" with no error message
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB

# ── 2. Load the Trained Model Once at Startup ────
# Use abspath to guarantee the correct path regardless of
# how gunicorn or python resolves __file__ at runtime
BASE_DIR   = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.h5')

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"model.h5 not found at {MODEL_PATH}. "
        "Make sure model.h5 is in the same folder as app.py."
    )

model = load_model(MODEL_PATH)
print(f"✅  model.h5 loaded successfully from {MODEL_PATH}")

# ── 3. Class Labels ───────────────────────────────
# image_dataset_from_directory assigns labels alphabetically:
#   cat/ → 0,  dog/ → 1
# sigmoid output:  >= 0.5 → Dog,  < 0.5 → Cat
LABELS = {0: 'Cat 🐱', 1: 'Dog 🐶'}

# Allowed image extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}

def allowed_file(filename: str) -> bool:
    """Return True only if the file has an accepted image extension."""
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    )

# ── 4. Global JSON Error Handlers ────────────────
# These catch crashes that happen OUTSIDE the try/except in predict(),
# e.g. a 413 when the uploaded file is too large.
# Without these, Flask returns an HTML page — breaking JSON.parse().
@app.errorhandler(413)
def file_too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 10 MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error. Check server logs.'}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Route not found.'}), 404

# ── 5. Home Route — Serve the HTML Page ──────────
@app.route('/')
def home():
    return render_template('index.html')

# ── 6. Prediction Route ───────────────────────────
@app.route('/predict', methods=['POST'])
def predict():

    # ── Check a file was included in the request ──
    if 'file' not in request.files:
        return jsonify({'error': 'No file field in request. '
                                 'Make sure FormData key is "file".'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    # ── Validate file extension ───────────────────
    if not allowed_file(file.filename):
        return jsonify({
            'error': f'Unsupported file type. '
                     f'Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 415

    try:
        # ── Read and open the image ───────────────
        img_bytes = file.read()

        if len(img_bytes) == 0:
            return jsonify({'error': 'Uploaded file is empty.'}), 400

        img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # ── Resize to match training input size ───
        # Your notebook used image_size=(200, 200) in
        # image_dataset_from_directory — this MUST match exactly.
        img_pil = img_pil.resize((200, 200))

        # ── Convert to numpy array ────────────────
        img_array = image.img_to_array(img_pil)   # shape: (200, 200, 3)

        # ── DO NOT normalise (/255.0) ─────────────
        # Your notebook never normalised pixel values during training.
        # image_dataset_from_directory loads raw pixel values (0–255).
        # Dividing by 255 here would cause a distribution mismatch
        # and produce wrong predictions every time.

        img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 200, 200, 3)

        # ── Run prediction ────────────────────────
        prediction = model.predict(img_array, verbose=0)
        raw_output = float(prediction[0][0])

        # ── Map sigmoid output to label ───────────
        if raw_output >= 0.5:
            label      = LABELS[1]                          # Dog
            confidence = round(raw_output * 100, 1)
        else:
            label      = LABELS[0]                          # Cat
            confidence = round((1 - raw_output) * 100, 1)

        return jsonify({
            'prediction': label,
            'confidence': f'{confidence}%',
            'raw_score':  round(raw_output, 4)
        })

    except Image.UnidentifiedImageError:
        # Pillow cannot read the file — it's corrupted or not a real image
        return jsonify({
            'error': 'Could not read the image. '
                     'The file may be corrupted or not a valid image.'
        }), 422

    except Exception as e:
        print(f"[ERROR] Prediction failed: {type(e).__name__}: {e}")
        return jsonify({'error': f'{type(e).__name__}: {str(e)}'}), 500


# ── 7. Entry Point ────────────────────────────────
if __name__ == '__main__':
    # Render passes the assigned port via the PORT environment variable.
    # Locally, this falls back to 5000.
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

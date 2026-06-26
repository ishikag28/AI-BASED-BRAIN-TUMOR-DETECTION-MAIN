import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template, request, send_from_directory
import keras
from keras.utils import load_img, img_to_array
import numpy as np
import os
#from gemini import validate_mri

# -------------------------------
# INIT APP
# -------------------------------
app = Flask(__name__)

# -------------------------------
# LOAD MODEL (FINAL WORKING)
# -------------------------------
import download_model

model = keras.models.load_model('models/model.h5', compile=False)

# -------------------------------
# CONFIG
# -------------------------------
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# -------------------------------
# PREPROCESS
# -------------------------------
def preprocess(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img = img_to_array(img) / 255.0
    return np.expand_dims(img, axis=0)

# -------------------------------
# PREDICTION
# -------------------------------
def predict_tumor(image_path):
    img = preprocess(image_path)
    preds = model.predict(img, verbose=0)

    idx = np.argmax(preds)
    confidence = float(np.max(preds))
    label = class_labels[idx]

    if label == "notumor":
        return "No Tumor", confidence
    else:
        return f"Tumor: {label}", confidence

# -------------------------------
# ROUTES
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")

        if not file:
            return render_template("index.html", error="No file uploaded")

        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        # Run prediction directly (skip Gemini validation)
        result, confidence = predict_tumor(path)

        return render_template(
            "index.html",
            result=result,
            confidence=round(confidence * 100, 2),
            file_path=f"/uploads/{file.filename}",
     )
    return render_template("index.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)

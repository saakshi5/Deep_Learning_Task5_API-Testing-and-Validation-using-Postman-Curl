from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load Pretrained MobileNetV2 Model
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# -----------------------------
# Home Page
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# API Status
# -----------------------------
@app.route("/api")
def api_status():
    return jsonify({
        "message": "Deep Learning Flask API",
        "model": "MobileNetV2",
        "status": "Running"
    })


# -----------------------------
# Prediction API
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:

        # Read image
        image = Image.open(file).convert("RGB")

        # Resize image
        image = image.resize((224, 224))

        # Convert to array
        img = np.array(image)
        img = np.expand_dims(img, axis=0)

        # Preprocess
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

        # Prediction
        pred = model.predict(img, verbose=0)

        # Decode Top 3 Predictions
        decoded = tf.keras.applications.mobilenet_v2.decode_predictions(
            pred,
            top=1
        )[0]

        results = []

        for item in decoded:
            results.append({
                "class": item[1].replace("_", " ").title(),
                "confidence": round(float(item[2]) * 100, 2)
            })

        return jsonify({
            "predictions": results
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
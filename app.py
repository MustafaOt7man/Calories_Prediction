from flask import Flask, request, jsonify
import joblib
import numpy as np
import traceback
import os

app = Flask(__name__)

# Load the trained GradientBoosting model
MODEL_PATH = "GradientBoosting_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
    print("✅ AI model loaded successfully.")
except Exception as e:
    print("❌ Failed to load model:", e)
    exit(1)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Extract expected input features from JSON body
        required_fields = [
            "totalSteps",
            "veryActiveMinutes",
            "fairlyActiveMinutes",
            "lightlyActiveMinutes",
            "sedentaryMinutes",
            "totalMinutesAsleep",
            "weightKg"
        ]

        if not all(field in data for field in required_fields):
            return jsonify({
                "success": False,
                "message": f"Missing fields in request. Required: {required_fields}",
                "predictedCalories": 0
            }), 400

        # Prepare input data
        features = np.array([
            data["totalSteps"],
            data["veryActiveMinutes"],
            data["fairlyActiveMinutes"],
            data["lightlyActiveMinutes"],
            data["sedentaryMinutes"],
            data["totalMinutesAsleep"],
            data["weightKg"]
        ]).reshape(1, -1)

        # Predict using the model
        prediction = model.predict(features)[0]

        return jsonify({
            "success": True,
            "message": "Prediction successful",
            "predictedCalories": float(prediction)
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error occurred: {str(e)}",
            "trace": traceback.format_exc(),
            "predictedCalories": 0
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Railway will inject PORT
    app.run(host="0.0.0.0", port=port)

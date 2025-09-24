from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Initialize Flask app
app = Flask(__name__)

# Home route for manual input + dropdown
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    if request.method == "POST":
        try:
            # Read form inputs
            input_data = [
                float(request.form["recency"]),
                float(request.form["frequency"]),
                float(request.form["monetary"]),
                float(request.form["time"])
            ]

            input_df = pd.DataFrame([input_data], columns=["Recency", "Frequency", "Monetary", "Time"])
            scaled_input = scaler.transform(input_df)

            probs = model.predict_proba(scaled_input)[0]
            threshold = 0.15
            pred = 1 if probs[1] > threshold else 0

            recommendation_map = {
                0: "✅ Healthy – No action needed",
                1: "📅 Schedule regular checkups"
            }

            confidence_pct = round(probs[pred] * 100, 2)
            prediction = f"{recommendation_map.get(pred)} (Confidence: {confidence_pct}%)"

        except Exception as e:
            prediction = f"❌ Error: {e}"

    return render_template("index.html", prediction=prediction)

# Test suite route
@app.route("/run-tests")
def run_tests():
    test_cases = [
        ("Very Healthy",            [5, 12, 600, 220], 0),
        ("Active Donor",            [12, 14, 550, 200], 0),
        ("Balanced Case",           [25, 6, 300, 120], "0 or 1"),
        ("Mid Risk",                [35, 4, 150, 90], "0 or 1"),
        ("High Recency Only",       [80, 8, 400, 200], 0),
        ("Low Frequency",           [20, 1, 350, 180], 1),
        ("Low Monetary & Time",     [15, 5, 50, 25], 1),
        ("High Risk Case 1",        [85, 1, 40, 15], 1),
        ("High Risk Case 2",        [90, 1, 20, 10], 1),
        ("Extreme Risk",            [95, 0, 10, 5], 1)
    ]

    results = []
    threshold = 0.15

    for name, features, expected in test_cases:
        df = pd.DataFrame([features], columns=["Recency", "Frequency", "Monetary", "Time"])
        scaled = scaler.transform(df)
        probs = model.predict_proba(scaled)[0]
        pred = 1 if probs[1] > threshold else 0
        confidence = round(probs[pred] * 100, 2)

        recommendation_map = {
            0: "✅ Healthy",
            1: "📅 Checkups"
        }

        result = recommendation_map.get(pred)
        status = "✅ PASS" if expected == pred or expected == "0 or 1" else "❌ FAIL"

        results.append({
            "name": name,
            "input": features,
            "prediction": result,
            "confidence": f"{confidence}%",
            "expected": expected,
            "status": status
        })

    return render_template("test_results.html", results=results)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, port=8080)


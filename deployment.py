from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('trained_models/ensemble_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        audio_file = request.files['audio']
        audio_path = f"temp_{audio_file.filename}"
        audio_file.save(audio_path)
        
        # Use your existing pipeline
        features = extract_features(audio_path)
        if features is None:
            return jsonify({"error": "Feature extraction failed"}), 400
            
        prediction = model.predict(features.values)[0]
        proba = model.predict_proba(features.values)[0][1]
        
        return jsonify({
            "prediction": "Depressed" if prediction == 1 else "Not Depressed",
            "confidence": float(proba),
            "risk_level": "High" if proba > 0.7 else ("Medium" if proba > 0.4 else "Low")
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
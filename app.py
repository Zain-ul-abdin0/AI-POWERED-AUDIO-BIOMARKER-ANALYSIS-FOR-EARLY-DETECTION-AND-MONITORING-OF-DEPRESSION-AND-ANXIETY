from flask import Flask, request, jsonify, render_template
import librosa
import numpy as np
import pandas as pd
import joblib
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained pipeline and features
MODEL_PATH = 'trained_models/ensemble_model.joblib'
FEATURE_NAMES_PATH = 'trained_models/overall_selected_features.joblib'

try:
    pipeline = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    print("✔ Model and features loaded successfully")
    print(f"Expected features ({len(feature_names)}): {feature_names}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    pipeline = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(audio_path):
    """Complete feature extraction matching training pipeline from main.py"""
    try:
        # Load and preprocess audio
        audio, sr = librosa.load(audio_path, sr=16000)
        audio = librosa.effects.preemphasis(audio)
        audio = librosa.util.normalize(audio)

        features = {}

        # 1. MFCC Features (40 coefficients)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        for i in range(20):  # As in main.py
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
            features[f'mfcc_{i}_kurtosis'] = pd.Series(mfccs[i]).kurtosis()

        # 2. Chroma Features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        for i in range(12):
            features[f'chroma_{i}_mean'] = np.mean(chroma[i])

        # 3. Spectral Features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        
        features.update({
            "spectral_centroid_mean": np.mean(spectral_centroid),
            "spectral_bandwidth_mean": np.mean(spectral_bandwidth),
            "spectral_rolloff_mean": np.mean(spectral_rolloff),
            "spectral_contrast_mean": np.mean(spectral_contrast),
        })

        # 4. Pitch and Voice Quality
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        valid_pitches = pitches[pitches > 0]
        pitch_mean = np.mean(valid_pitches) if len(valid_pitches) > 0 else 0
        
        features.update({
            "pitch_mean": pitch_mean,
            "pitch_std": np.std(valid_pitches) if len(valid_pitches) > 0 else 0,
            "harmonic_ratio": np.mean(librosa.effects.harmonic(audio)),
            "percussive_ratio": np.mean(librosa.effects.percussive(audio)),
        })

        # 5. Voice Activity Detection
        rms = librosa.feature.rms(y=audio)
        vad = rms > np.percentile(rms, 30)
        features.update({
            'vad_ratio': np.mean(vad),
            'vad_transitions': np.sum(np.diff(vad.astype(int)) != 0)
        })

        # 6. Jitter/Shimmer
        pitches = librosa.yin(audio, fmin=80, fmax=400)
        valid_pitches = pitches[pitches > 0]
        if len(valid_pitches) > 1:
            features['jitter'] = np.mean(np.abs(np.diff(valid_pitches))) / np.mean(valid_pitches)
        else:
            features['jitter'] = 0

        # 7. Speaking Rate/Pauses
        intervals = librosa.effects.split(audio, top_db=25)
        speech_duration = sum(end - start for start, end in intervals) / sr
        total_duration = len(audio) / sr
        features.update({
            "pause_ratio": (total_duration - speech_duration) / total_duration if total_duration > 0 else 0,
            "speaking_rate": len(intervals) / total_duration if total_duration > 0 else 0,
        })

        # Create DataFrame and ensure all expected features exist
        df = pd.DataFrame([features])
        
        # Add missing features with default 0
        for feat in feature_names:
            if feat not in df.columns:
                df[feat] = 0.0
                print(f"⚠ Added missing feature: {feat}")

        return df[feature_names]  # Return in exact expected order

    except Exception as e:
        print(f"Feature extraction failed: {str(e)}")
        raise

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
        
    if not (file and allowed_file(file.filename)):
        return jsonify({"error": "Invalid file type"}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Verify audio duration
        duration = librosa.get_duration(filename=filepath)
        if duration < 3:
            os.remove(filepath)
            return jsonify({"error": f"Audio too short ({duration:.1f}s). Minimum 3 seconds required."}), 400

        features = extract_features(filepath)
        os.remove(filepath)  # Clean up

        if pipeline is None:
            return jsonify({"error": "Model not loaded"}), 500

        proba = pipeline.predict_proba(features.values)[0][1]
        prediction = "Depressed" if proba > 0.5 else "Not Depressed"
        
        return jsonify({
            "prediction": prediction,
            "confidence": float(proba),
            "risk_level": "High" if proba > 0.7 else ("Medium" if proba > 0.4 else "Low"),
            "features": {k: float(v) for k, v in features.iloc[0].items()}
        })

    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
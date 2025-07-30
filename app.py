from flask import Flask, request, jsonify, render_template
import librosa
import numpy as np
import pandas as pd
import joblib
import os
import soundfile as sf
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained pipeline and features
MODEL_PATH = 'trained_models/xgboost_model.joblib'
FEATURE_NAMES_PATH = 'trained_models/xgboost_features.joblib'

try:
    pipeline = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    print("✔ Model and features loaded successfully")
    print(f"Model expects {len(feature_names)} features")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    pipeline = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_audio(audio_path, target_sr=16000, duration=5):
    """EXACT replica from your main.py"""
    try:
        audio, sr = librosa.load(audio_path, sr=target_sr)
        
        # Fixed duration handling
        if len(audio) < target_sr * duration:
            audio = np.pad(audio, (0, max(0, int(target_sr * duration) - len(audio))), 'constant')
        else:
            audio = audio[:int(target_sr * duration)]
        
        # Trim silence (same parameters)
        audio, _ = librosa.effects.trim(audio, top_db=25)
        
        if len(audio) < sr * 0.5:
            raise ValueError("Audio too short after trimming")
            
        audio = librosa.effects.preemphasis(audio)
        audio = librosa.util.normalize(audio)
        return audio, sr
        
    except Exception as e:
        print(f"Audio processing error: {e}")
        raise

def extract_features(audio_path):
    """IDENTICAL to LoadModel.py's extraction"""
    features = {}
    
    # 1. Process audio (must match exactly)
    audio, sr = process_audio(audio_path)
    
    # 2. MFCC Features (40 coefficients, 20 with stats)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    for i in range(20):
        features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
        features[f'mfcc_{i}_std'] = np.std(mfccs[i])
        features[f'mfcc_{i}_kurtosis'] = pd.Series(mfccs[i]).kurtosis()
    
    # 3. Chroma Features
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    for i in range(12):
        features[f'chroma_{i}_mean'] = np.mean(chroma[i])
    
    # 4. Spectral Features
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
    
    # 5. Pitch and Voice Quality
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    valid_pitches = pitches[pitches > 0]
    pitch_mean = np.mean(valid_pitches) if len(valid_pitches) > 0 else 0
    
    features.update({
        "pitch_mean": pitch_mean,
        "pitch_std": np.std(valid_pitches) if len(valid_pitches) > 0 else 0,
        "harmonic_ratio": np.mean(librosa.effects.harmonic(audio)),
        "percussive_ratio": np.mean(librosa.effects.percussive(audio)),
    })
    
    # 6. Voice Activity Detection (must match exactly)
    rms = librosa.feature.rms(y=audio)
    vad = rms > np.percentile(rms, 30)  # Same 30th percentile
    features.update({
        'vad_ratio': np.mean(vad),
        'vad_transitions': np.sum(np.diff(vad.astype(int)) != 0)
    })
    
    # 7. Jitter Calculation
    pitches = librosa.yin(audio, fmin=80, fmax=400)
    valid_pitches = pitches[pitches > 0]
    if len(valid_pitches) > 1:
        features['jitter'] = np.mean(np.abs(np.diff(valid_pitches))) / np.mean(valid_pitches)
    else:
        features['jitter'] = 0
    
    # 8. Pause Statistics
    intervals = librosa.effects.split(audio, top_db=25)  # Same threshold
    speech_duration = sum(end - start for start, end in intervals) / sr
    total_duration = len(audio) / sr
    features.update({
        "pause_ratio": (total_duration - speech_duration) / total_duration if total_duration > 0 else 0,
        "speaking_rate": len(intervals) / total_duration if total_duration > 0 else 0,
        "speech_rate_var": np.std([(end-start)/sr for start,end in intervals]) if len(intervals) > 1 else 0,
    })
    
    # Create DataFrame with exact feature order
    df = pd.DataFrame([features])
    
    # Debug: Verify feature counts
    print(f"Extracted {len(features)} features, model expects {len(feature_names)}")
    missing = set(feature_names) - set(features.keys())
    if missing:
        print(f"⚠ Adding {len(missing)} missing features with 0 values")
        for feat in missing:
            df[feat] = 0.0
    
    return df[feature_names]  # Critical: same order as training

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
        
    if not allowed_file(file.filename):
        return jsonify({"error": "Only WAV/MP3 files allowed"}), 400

    try:
        # Save temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Verify duration
        duration = librosa.get_duration(filename=filepath)
        if duration < 3:
            raise ValueError(f"Audio too short ({duration:.2f}s). Need ≥3s.")
        
        # Extract features (identical to LoadModel.py)
        features = extract_features(filepath)
        os.remove(filepath)  # Clean up
        
        # Predict
        proba = pipeline.predict_proba(features.values)[0][1]
        confidence = round(proba * 100, 1)
        
        return jsonify({
            "prediction": "Depressed" if proba > 0.5 else "Not Depressed",
            "confidence": confidence,
            "risk_level": "High" if proba > 0.7 else ("Medium" if proba > 0.4 else "Low"),
            "features": {k: float(v) for k, v in features.iloc[0].items() 
                        if k in ['mfcc_0_mean', 'jitter', 'pause_ratio']}  # Sample features
        })
        
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
import librosa
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler

# Define feature order matching training
FEATURE_ORDER = [
    'mfcc_0', 'mfcc_1', 'mfcc_2',
    'chroma_0', 'chroma_1',
    'spectral_centroid',
    'spectral_bandwidth',
    'pitch_mean',
    'pause_ratio',
    'speech_rate',
    'zcr_mean',
    'rms_energy'
]

def load_model_pipeline():
    """Load the complete trained pipeline"""
    try:
        pipeline = joblib.load('./svm_model.joblib')
        print("✔ Full model pipeline loaded successfully")
        return pipeline
    except Exception as e:
        print(f"× Error loading model pipeline: {str(e)}")
        return None

def extract_features(file_path):
    """Corrected feature extraction with validation"""
    try:
        y, sr = librosa.load(file_path, sr=22050)
        
        # Validate audio isn't silent
        if np.max(np.abs(y)) < 0.01:
            print("× Audio is too quiet/silent")
            return None
            
        # Recalculate features with proper methods
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Calculate pauses properly
        non_silent = librosa.effects.split(y, top_db=30)
        pause_ratio = sum(end-start for (start,end) in non_silent)/len(y)
        
        features = {
            'mfcc_0': np.mean(mfcc[0]),
            'mfcc_1': np.mean(mfcc[1]),
            'mfcc_2': np.mean(mfcc[2]),
            'chroma_0': np.mean(chroma[0]),
            'chroma_1': np.mean(chroma[1]),
            'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            'spectral_bandwidth': np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
            'pitch_mean': np.median(librosa.yin(y, fmin=80, fmax=400)),
            'pause_ratio': pause_ratio,
            'speech_rate': len(non_silent)/(len(y)/sr),
            'zcr_mean': np.mean(librosa.feature.zero_crossing_rate(y)),
            'rms_energy': np.mean(librosa.feature.rms(y=y))
        }
        
        # Convert to DataFrame with correct column order
        return pd.DataFrame([features])[FEATURE_ORDER]
        
    except Exception as e:
        print(f"Feature extraction failed: {str(e)}")
        return None

def predict_audio(file_path, pipeline):
    """Make prediction using full pipeline"""
    if not os.path.exists(file_path):
        print(f"× File not found: {file_path}")
        return "Error: File not found", 0.0
    
    features = extract_features(file_path)
    if features is None:
        return "Error: Feature extraction failed", 0.0
    
    try:
        # Convert to numpy array to avoid feature name warnings
        features_array = features.values
        
        # Make prediction
        pred = pipeline.predict(features_array)
        proba = pipeline.predict_proba(features_array)
        
        label = "Depressed" if pred[0] == 1 else "Not Depressed"
        confidence = proba[0][pred[0]] * 100
        return label, round(confidence, 1)
    
    except Exception as e:
        print(f"Prediction failed: {str(e)}")
        return "Error: Prediction failed", 0.0

if __name__ == "__main__":
    pipeline = load_model_pipeline()
    if pipeline is None:
        exit()
    
    audio_file = "./Test/491_AUDIO.wav"
    label, confidence = predict_audio(audio_file, pipeline)
    
    print("\n" + "="*50)
    print(f"  Result: {label} ({confidence}% confidence)")
    print("="*50)
import librosa
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler

# 1. Model Loading with Error Handling
def load_model_components():
    """Load model components with validation"""
    components = {}
    try:
        components['model'] = joblib.load('./svm_model.joblib')
        print("✔ SVM model loaded successfully")
    except Exception as e:
        print(f"× Error loading model: {str(e)}")
        return None

    # Handle scaler (optional but recommended)
    if os.path.exists('./scaler.joblib'):
        try:
            components['scaler'] = joblib.load('./scaler.joblib')
            print("✔ Scaler loaded successfully")
        except Exception as e:
            print(f"× Error loading scaler: {str(e)}")
            components['scaler'] = None
    else:
        print("⚠ No scaler found - proceeding without scaling")
        components['scaler'] = None

    return components

# 2. Enhanced Feature Extraction (9 features)
def extract_features(file_path):
    """Extract exactly 9 features matching training data"""
    try:
        y, sr = librosa.load(file_path, sr=22050)  # Fixed sample rate
        
        # Feature calculations
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        features = {
            'mfcc_mean': np.mean(mfcc),
            'mfcc_std': np.std(mfcc),
            'chroma_mean': np.mean(chroma),
            'chroma_std': np.std(chroma),
            'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            'spectral_bandwidth': np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
            'pitch_mean': np.mean(librosa.yin(y, fmin=50, fmax=2000)),
            'pause_ratio': len(librosa.effects.split(y, top_db=30)) / len(y),
            'speech_rate': np.count_nonzero(y > 0.1*np.max(y))/len(y)
        }
        
        # Ensure consistent feature order
        feature_order = [
            'mfcc_mean', 'mfcc_std',
            'chroma_mean', 'chroma_std',
            'spectral_centroid',
            'spectral_bandwidth',
            'pitch_mean',
            'pause_ratio',
            'speech_rate'
        ]
        return pd.DataFrame([features])[feature_order]
    
    except Exception as e:
        print(f"Feature extraction failed: {str(e)}")
        return None

# 3. Robust Prediction Function
def predict_audio(file_path, components):
    """Make prediction with error handling"""
    if not os.path.exists(file_path):
        print(f"× File not found: {file_path}")
        return "Error: File not found", 0.0
    
    features = extract_features(file_path)
    if features is None:
        return "Error: Feature extraction failed", 0.0
    
    try:
        # Scale if scaler exists
        if components['scaler'] is not None:
            features = components['scaler'].transform(features)
        
        # Predict
        pred = components['model'].predict(features)
        proba = components['model'].predict_proba(features)
        
        label = "Depressed" if pred[0] == 1 else "Not Depressed"
        confidence = proba[0][pred[0]] * 100
        return label, round(confidence, 1)
    
    except Exception as e:
        print(f"Prediction failed: {str(e)}")
        return "Error: Prediction failed", 0.0

# 4. Main Execution
if __name__ == "__main__":
    # Load components
    model_components = load_model_components()
    if model_components is None:
        print("Critical error - cannot proceed")
        exit()
    
    # Example prediction
    audio_file = "./Dataset/307_AUDIO.wav"  # Replace with your file
    label, confidence = predict_audio(audio_file, model_components)
    
    print("\n" + "="*50)
    print(f"  Result: {label} ({confidence}% confidence)")
    print("="*50)
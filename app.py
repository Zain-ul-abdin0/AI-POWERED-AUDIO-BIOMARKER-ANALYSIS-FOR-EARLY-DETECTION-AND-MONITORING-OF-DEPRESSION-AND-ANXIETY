# # Import necessary libraries
# from flask import Flask, request, jsonify, render_template
# import librosa
# import numpy as np
# import pandas as pd
# import joblib
# import os
# from werkzeug.utils import secure_filename
# import warnings

# # Suppress librosa's 'audioread' warning which can be noisy
# warnings.filterwarnings("ignore", category=UserWarning)

# # Initialize Flask app
# app = Flask(__name__)

# # --- Configuration ---
# UPLOAD_FOLDER = 'uploads'
# ALLOWED_EXTENSIONS = {'wav', 'mp3'}
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # A lower prediction threshold can help detect the minority class (Depressed) more effectively.
# PREDICTION_THRESHOLD = 0.4 
# HIGH_RISK_THRESHOLD = 0.6
# MEDIUM_RISK_THRESHOLD = 0.3

# # --- Load Model and Feature Names ---
# # IMPORTANT: These files must exist in a './trained_models' directory.
# MODEL_PATH = './trained_models/random_forest_model.joblib'
# FEATURE_NAMES_PATH = './trained_models/overall_selected_features.joblib'

# try:
#     pipeline = joblib.load(MODEL_PATH)
#     feature_names = joblib.load(FEATURE_NAMES_PATH)
#     print("✔ Model and features loaded successfully")
#     print(f"Model expects {len(feature_names)} features")
# except Exception as e:
#     print(f"❌ Error loading model: {e}")
#     print("Please ensure the model and features files are in the './trained_models' directory.")
#     pipeline = None
#     feature_names = []  # Initialize as empty list to prevent runtime errors

# def allowed_file(filename):
#     """Check if the uploaded file has an allowed extension."""
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def process_audio(audio_path, target_sr=16000, duration=5):
#     """
#     Processes the audio file. This function is an exact replica of the
#     one used in `main.py` to ensure consistency. It pads/trims audio to
#     a fixed duration of 5 seconds.
#     """
#     try:
#         # Load audio with target sample rate
#         audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        
#         # Trim or pad audio to a fixed duration
#         if len(audio) < target_sr * duration:
#             audio = np.pad(audio, (0, max(0, int(target_sr * duration) - len(audio))), 'constant')
#         else:
#             audio = audio[:int(target_sr * duration)]
            
#         # Remove silence (less aggressive than before)
#         audio, _ = librosa.effects.trim(audio, top_db=25)
        
#         # Check for sufficient audio after trimming
#         if len(audio) < sr * 0.5:
#             raise ValueError("Audio too short after trimming. Must be at least 0.5 seconds.")
            
#         # Apply pre-emphasis and normalize the audio
#         audio = librosa.effects.preemphasis(audio)
#         audio = librosa.util.normalize(audio)
        
#         return audio, sr
    
#     except Exception as e:
#         print(f"Audio processing error: {e}")
#         raise

# def extract_features(audio_path):
#     """
#     Extracts features exactly as it is done in `main.py` to ensure
#     the input to the model is consistent with its training data.
#     """
#     features = {}
    
#     # 1. Process audio with a fixed duration of 5 seconds
#     audio, sr = process_audio(audio_path, duration=5)
    
#     # 2. MFCC Features (40 coefficients, but with stats for only the first 20)
#     # This is critical to match the model training process.
#     n_mfcc_to_use = 20
#     mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
#     for i in range(n_mfcc_to_use):
#         features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
#         features[f'mfcc_{i}_std'] = np.std(mfccs[i])
#         features[f'mfcc_{i}_kurtosis'] = pd.Series(mfccs[i]).kurtosis()
    
#     # 3. Chroma Features
#     chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
#     for i in range(12):
#         features[f'chroma_{i}_mean'] = np.mean(chroma[i])
    
#     # 4. Spectral Features
#     spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
#     spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
#     spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
#     spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    
#     features.update({
#         "spectral_centroid_mean": np.mean(spectral_centroid),
#         "spectral_bandwidth_mean": np.mean(spectral_bandwidth),
#         "spectral_rolloff_mean": np.mean(spectral_rolloff),
#         "spectral_contrast_mean": np.mean(spectral_contrast),
#     })
    
#     # 5. Pitch and Voice Quality
#     pitches, _ = librosa.core.piptrack(y=audio, sr=sr)
#     valid_pitches = pitches[pitches > 0]
#     pitch_mean = np.mean(valid_pitches) if len(valid_pitches) > 0 else 0
    
#     features.update({
#         "pitch_mean": pitch_mean,
#         "pitch_std": np.std(valid_pitches) if len(valid_pitches) > 0 else 0,
#         "harmonic_ratio": np.mean(librosa.effects.harmonic(audio)),
#         "percussive_ratio": np.mean(librosa.effects.percussive(audio)),
#     })
    
#     # 6. Voice Activity Detection
#     rms = librosa.feature.rms(y=audio)
#     vad = rms > np.percentile(rms, 30)
#     features.update({
#         'vad_ratio': np.mean(vad),
#         'vad_transitions': np.sum(np.diff(vad.astype(int)) != 0)
#     })
    
#     # 7. Jitter Calculation
#     pitches_yin = librosa.yin(audio, fmin=80, fmax=400)
#     valid_pitches_yin = pitches_yin[pitches_yin > 0]
#     if len(valid_pitches_yin) > 1:
#         features['jitter'] = np.mean(np.abs(np.diff(valid_pitches_yin))) / np.mean(valid_pitches_yin)
#     else:
#         features['jitter'] = 0
    
#     # 8. Pause Statistics
#     intervals = librosa.effects.split(audio, top_db=25)
#     speech_duration = sum(end - start for start, end in intervals) / sr
#     total_duration = len(audio) / sr
#     features.update({
#         "pause_ratio": (total_duration - speech_duration) / total_duration if total_duration > 0 else 0,
#         "speaking_rate": len(intervals) / total_duration if total_duration > 0 else 0,
#         "speech_rate_var": np.std([(end-start)/sr for start,end in intervals]) if len(intervals) > 1 else 0,
#     })
    
#     # Create DataFrame and ensure features are in the correct order
#     df = pd.DataFrame([features])
    
#     # Critical: Use the `feature_names` list to enforce the correct order and handle
#     # any missing features, which should no longer be an issue with this corrected
#     # extraction logic.
    
#     # Extract only the feature names if the loaded object is a list of tuples
#     if isinstance(feature_names[0], tuple):
#         expected_feature_names = [name for name, _ in feature_names]
#     else:
#         expected_feature_names = feature_names

#     # Handle any missing features by adding them with a value of 0.0
#     missing = set(expected_feature_names) - set(features.keys())
#     if missing:
#         print(f"⚠ Adding {len(missing)} missing features with 0 values")
#         for feat in missing:
#             df[feat] = 0.0
    
#     return df[expected_feature_names]

# # --- Routes ---
# @app.route('/', methods=['GET'])
# def home():
#     """Render the homepage template."""
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     """Handle file upload, feature extraction, and prediction."""
#     if not pipeline:
#         return jsonify({"error": "Model not loaded. Please check the server logs."}), 500

#     if 'file' not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400
    
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "Empty filename"}), 400
    
#     if not allowed_file(file.filename):
#         return jsonify({"error": "Only WAV or MP3 files allowed"}), 400

#     filepath = None
#     try:
#         # Save the file temporarily
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)

#         # Extract features
#         features = extract_features(filepath)
        
#         # Predict probability
#         proba = pipeline.predict_proba(features.values)[0][1]
#         confidence = round(proba * 100, 1)
        
#         # Determine prediction and risk level using the new thresholds
#         prediction = "Depressed" if proba > PREDICTION_THRESHOLD else "Not Depressed"
#         risk_level = "High" if proba > HIGH_RISK_THRESHOLD else ("Medium" if proba > MEDIUM_RISK_THRESHOLD else "Low")
        
#         # Prepare response
#         response_features = {
#             k: float(v) for k, v in features.iloc[0].items()
#             if k in ['mfcc_0_mean', 'jitter', 'pause_ratio']
#         }
        
#         return jsonify({
#             "prediction": prediction,
#             "confidence": confidence,
#             "risk_level": risk_level,
#             "features": response_features
#         })
        
#     except ValueError as ve:
#         # Handle audio-specific errors gracefully
#         return jsonify({"error": str(ve)}), 400
    
#     except Exception as e:
#         # Catch any other unexpected errors
#         print(f"An error occurred: {e}")
#         return jsonify({"error": f"An unexpected error occurred: {e}"}), 500
    
#     finally:
#         # Clean up the uploaded file
#         if filepath and os.path.exists(filepath):
#             os.remove(filepath)

# if __name__ == '__main__':
#     # Ensure all required directories exist before running
#     if not os.path.exists('./trained_models'):
#         print("Creating './trained_models' directory.")
#         os.makedirs('./trained_models')
#         print("Please place your 'random_forest_model.joblib' and 'random_forest_features.joblib' files here.")
#     app.run(host='0.0.0.0', port=5000, debug=True)




# Import necessary libraries
from flask import Flask, request, jsonify, render_template
import librosa
import numpy as np
import pandas as pd
import joblib
import os
from werkzeug.utils import secure_filename
import warnings

# Suppress librosa's 'audioread' warning which can be noisy
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize Flask app
app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3'}
MODEL_SAVE_PATH = 'trained_models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# A lower prediction threshold can help detect the minority class (Depressed) more effectively.
PREDICTION_THRESHOLD = 0.4
HIGH_RISK_THRESHOLD = 0.6
MEDIUM_RISK_THRESHOLD = 0.3

# --- Model and Feature File Paths ---
# This dictionary maps the model name from the dropdown to its corresponding model file path.
# All models will use the same overall_selected_features.joblib file as per your request.
MODEL_PATHS = {
    'ensemble': os.path.join(MODEL_SAVE_PATH, 'ensemble_model.joblib'),
    'lightgbm': os.path.join(MODEL_SAVE_PATH, 'lightgbm_model.joblib'),
    'random_forest': os.path.join(MODEL_SAVE_PATH, 'random_forest_model.joblib'),
    'svm': os.path.join(MODEL_SAVE_PATH, 'svm_model.joblib'),
    'xgboost': os.path.join(MODEL_SAVE_PATH, 'xgboost_model.joblib'),
}
FEATURE_NAMES_PATH = os.path.join(MODEL_SAVE_PATH, 'overall_selected_features.joblib')

# Attempt to load the shared feature names file once at startup
try:
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    print("✔ Feature names loaded successfully.")
    print(f"All models will use {len(feature_names)} features.")
except Exception as e:
    print(f"❌ Error loading feature names: {e}")
    print("Please ensure 'overall_selected_features.joblib' is in the './trained_models' directory.")
    feature_names = [] # Fallback to an empty list to prevent runtime errors

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_audio(audio_path, target_sr=16000, duration=5):
    """
    Processes the audio file. This function is an exact replica of the
    one used in `main.py` to ensure consistency. It pads/trims audio to
    a fixed duration of 5 seconds.
    """
    try:
        # Load audio with target sample rate
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        
        # Trim or pad audio to a fixed duration
        if len(audio) < target_sr * duration:
            audio = np.pad(audio, (0, max(0, int(target_sr * duration) - len(audio))), 'constant')
        else:
            audio = audio[:int(target_sr * duration)]
            
        # Remove silence (less aggressive than before)
        audio, _ = librosa.effects.trim(audio, top_db=25)
        
        # Check for sufficient audio after trimming
        if len(audio) < sr * 0.5:
            raise ValueError("Audio too short after trimming. Must be at least 0.5 seconds.")
            
        # Apply pre-emphasis and normalize the audio
        audio = librosa.effects.preemphasis(audio)
        audio = librosa.util.normalize(audio)
        
        return audio, sr
    
    except Exception as e:
        print(f"Audio processing error: {e}")
        raise

def extract_features(audio_path, feature_names):
    """
    Extracts features exactly as it is done in `main.py` to ensure
    the input to the model is consistent with its training data.
    
    This function now takes feature_names as an argument to align the
    dataframe correctly with the dynamically loaded model.
    """
    features = {}
    
    # 1. Process audio with a fixed duration of 5 seconds
    audio, sr = process_audio(audio_path, duration=5)
    
    # 2. MFCC Features (40 coefficients, but with stats for only the first 20)
    # This is critical to match the model training process.
    n_mfcc_to_use = 20
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    for i in range(n_mfcc_to_use):
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
    pitches, _ = librosa.core.piptrack(y=audio, sr=sr)
    valid_pitches = pitches[pitches > 0]
    pitch_mean = np.mean(valid_pitches) if len(valid_pitches) > 0 else 0
    
    features.update({
        "pitch_mean": pitch_mean,
        "pitch_std": np.std(valid_pitches) if len(valid_pitches) > 0 else 0,
        "harmonic_ratio": np.mean(librosa.effects.harmonic(audio)),
        "percussive_ratio": np.mean(librosa.effects.percussive(audio)),
    })
    
    # 6. Voice Activity Detection
    rms = librosa.feature.rms(y=audio)
    vad = rms > np.percentile(rms, 30)
    features.update({
        'vad_ratio': np.mean(vad),
        'vad_transitions': np.sum(np.diff(vad.astype(int)) != 0)
    })
    
    # 7. Jitter Calculation
    pitches_yin = librosa.yin(audio, fmin=80, fmax=400)
    valid_pitches_yin = pitches_yin[pitches_yin > 0]
    if len(valid_pitches_yin) > 1:
        features['jitter'] = np.mean(np.abs(np.diff(valid_pitches_yin))) / np.mean(valid_pitches_yin)
    else:
        features['jitter'] = 0
    
    # 8. Pause Statistics
    intervals = librosa.effects.split(audio, top_db=25)
    speech_duration = sum(end - start for start, end in intervals) / sr
    total_duration = len(audio) / sr
    features.update({
        "pause_ratio": (total_duration - speech_duration) / total_duration if total_duration > 0 else 0,
        "speaking_rate": len(intervals) / total_duration if total_duration > 0 else 0,
        "speech_rate_var": np.std([(end-start)/sr for start,end in intervals]) if len(intervals) > 1 else 0,
    })
    
    # Create DataFrame and ensure features are in the correct order
    df = pd.DataFrame([features])
    
    # The loaded feature names list might be a list of tuples, so we handle that case.
    if feature_names and isinstance(feature_names[0], tuple):
        expected_feature_names = [name for name, _ in feature_names]
    else:
        expected_feature_names = feature_names

    # Handle any missing features by adding them with a value of 0.0
    missing = set(expected_feature_names) - set(features.keys())
    if missing:
        print(f"⚠ Adding {len(missing)} missing features with 0 values")
        for feat in missing:
            df[feat] = 0.0
    
    return df[expected_feature_names]

# --- Routes ---
@app.route('/', methods=['GET'])
def home():
    """Renders the HTML homepage from the 'templates' directory."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the audio file upload and returns a depression risk prediction.
    It now dynamically loads the model based on the user's selection.
    """
    # 1. Get the model name from the form data
    model_name = request.form.get('model_name')
    if model_name not in MODEL_PATHS:
        return jsonify({"error": "Invalid model selected."}), 400

    # 2. Dynamically load the correct model
    try:
        model_file = MODEL_PATHS[model_name]
        pipeline = joblib.load(model_file)
        
    except FileNotFoundError:
        return jsonify({"error": f"Model file for '{model_name}' not found. Please ensure the files exist in the './trained_models' directory."}), 404

    # Check if a file was uploaded
    if 'file' not in request.files or not request.files['file'].filename:
        return jsonify({"error": "No file part or no selected file"}), 400
    
    file = request.files['file']
    filepath = None
    
    if file and allowed_file(file.filename):
        try:
            # Save the file temporarily
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Extract features from the uploaded audio file using the shared feature names
            if not feature_names:
                 return jsonify({"error": "Feature names not loaded. Cannot extract features."}), 500
                 
            features = extract_features(filepath, feature_names)
            
            # Predict probability of the 'Depressed' class
            proba = pipeline.predict_proba(features.values)[0][1]
            confidence = round(proba * 100, 1)
            
            # Determine prediction and risk level using the new thresholds
            prediction = "Depressed" if proba > PREDICTION_THRESHOLD else "Not Depressed"
            risk_level = "High" if proba > HIGH_RISK_THRESHOLD else ("Medium" if proba > MEDIUM_RISK_THRESHOLD else "Low")
            
            # Prepare response
            response_features = {
                k: float(v) for k, v in features.iloc[0].items()
                if k in ['mfcc_0_mean', 'jitter', 'pause_ratio']
            }
            
            return jsonify({
                "prediction": prediction,
                "confidence": confidence,
                "risk_level": risk_level,
                "features": response_features
            })
            
        except ValueError as ve:
            # Handle audio-specific errors gracefully
            return jsonify({"error": str(ve)}), 400
        
        except Exception as e:
            # Catch any other unexpected errors
            print(f"An error occurred: {e}")
            return jsonify({"error": f"An unexpected error occurred: {e}"}), 500
        
        finally:
            # Clean up the uploaded file
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
    else:
        return jsonify({"error": "File type not allowed"}), 400

if __name__ == '__main__':
    # Ensure all required directories exist before running
    if not os.path.exists('./trained_models'):
        print("Creating './trained_models' directory.")
        os.makedirs('./trained_models')
        print("Please place your trained model and feature files in this directory.")
    app.run(host='0.0.0.0', port=5000, debug=True)

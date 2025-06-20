# import librosa
# import numpy as np
# import pandas as pd
# import joblib
# import os
# from sklearn.preprocessing import StandardScaler
# # from textblob import TextBlob # Uncomment if your SELECTED_FEATURES include linguistic features and you have transcripts

# # Define the EXACT 30 features your model expects.
# # This list is CRITICAL and must precisely match the features (and their order)
# # that your 'svm_model.joblib' was trained on after feature selection in main.py.
# SELECTED_FEATURES = [
#     'mfcc_0_mean', 'mfcc_1_mean', 'mfcc_2_mean', 'mfcc_3_mean', 'mfcc_4_mean',
#     'mfcc_5_mean', 'mfcc_6_mean', 'mfcc_7_mean', 'mfcc_8_mean', 'mfcc_9_mean',
#     'chroma_0_mean', 'chroma_1_mean', 'chroma_2_mean', 'chroma_3_mean', 'chroma_4_mean',
#     'spectral_centroid_mean', 'spectral_bandwidth_mean', 'spectral_rolloff_mean',
#     'pitch_mean', 'pause_ratio', 'harmonic_ratio', 'percussive_ratio',
#     'vad_ratio', 'vad_transitions', 'jitter',
#     'speaking_rate', 'speech_rate_var',
#     'mfcc_0_std', 'mfcc_1_std', 'mfcc_2_std'
# ]

# def load_model_pipeline():
#     """Load the complete trained pipeline"""
#     try:
#         pipeline = joblib.load('./trained_models/svm_model.joblib')
#         print("✔ Full model pipeline loaded successfully")
        
#         # Debug: If your pipeline itself has a SelectKBest step (e.g., if you saved
#         # the entire make_pipeline including SelectKBest), you might be able to
#         # inspect its expected features. However, based on your main.py, the
#         # SelectKBest was applied *before* the SVM pipeline was trained.
#         # So, the 'SELECTED_FEATURES' list above is the most direct way to specify.
#         # print("Model expects these features:")
#         # print(pipeline.named_steps['selectkbest'].get_feature_names_out())
        
#         return pipeline
#     except Exception as e:
#         print(f"× Error loading model pipeline: {str(e)}")
#         return None

# def process_audio(audio_path, target_sr, duration):
#     """
#     Processes audio by loading, trimming/padding, silence removal,
#     normalization, and pre-emphasis. Matches logic from main.py.
#     """
#     try:
#         audio, sr = librosa.load(audio_path, sr=target_sr)

#         # Trim or pad audio to fixed duration
#         # Ensure integer for slicing/padding calculation
#         if len(audio) < target_sr * duration:
#             audio = np.pad(audio, (0, max(0, int(target_sr * duration) - len(audio))), 'constant')
#         else:
#             audio = audio[:int(target_sr * duration)]

#         # Remove silence (less aggressive than before)
#         audio, _ = librosa.effects.trim(audio, top_db=25)

#         # Check for sufficient audio after trimming
#         if len(audio) < sr * 0.5: # Require at least 0.5 seconds of audio
#             print(f"Audio too short after trimming: {audio_path}")
#             return None, None
        
#         # Enhanced normalization and pre-emphasis
#         audio = librosa.effects.preemphasis(audio)
#         audio = librosa.util.normalize(audio)
        
#         return audio, sr

#     except Exception as e:
#         print(f"An error occurred during audio processing for {audio_path}: {e}")
#         return None, None

# def voice_activity_features(audio, sr):
#     """
#     Calculate voice activity detection features.
#     Copied directly from main.py to ensure consistency.
#     """
#     rms = librosa.feature.rms(y=audio)
#     vad = rms > np.percentile(rms, 30)
#     return {
#         'vad_ratio': np.mean(vad),
#         'vad_transitions': np.sum(np.diff(vad.astype(int)) != 0) # Fixed parenthesis here
#     }

# def calculate_jitter(audio, sr):
#     """
#     Helper function for jitter calculation.
#     Copied from main.py's jitter_shimmer logic.
#     """
#     pitches = librosa.yin(audio, fmin=80, fmax=400)
#     valid_pitches = pitches[pitches > 0]
#     if len(valid_pitches) > 1:
#         return np.mean(np.abs(np.diff(valid_pitches))) / np.mean(valid_pitches)
#     return 0

# def extract_features(file_path):
#     """
#     Extracts all *potential* features that were generated in main.py's
#     extract_features function, then selects only the features specified
#     in SELECTED_FEATURES.
#     """
#     features = {}
    
#     # Process audio (matches the first part of extract_features in main.py)
#     audio, sr = process_audio(file_path, target_sr=16000, duration=5)
#     if audio is None:
#         return None

#     # --- Acoustic Features ---
#     # MFCCs: main.py uses n_mfcc=40 and extracts mean, std, kurtosis for first 20.
#     # To cover SELECTED_FEATURES, we need to ensure enough MFCCs are computed
#     # and then extract the mean/std/kurtosis for the relevant indices.
#     mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40) # Compute enough to cover needed MFCCs
    
#     # Extract means, stds, and kurtosis for MFCCs
#     # Loop up to 20 to cover mfcc_0_std to mfcc_19_std/kurtosis if needed
#     for i in range(20): 
#         features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
#         features[f'mfcc_{i}_std'] = np.std(mfccs[i])
#         # Only add kurtosis if you expect it in SELECTED_FEATURES and it was used in training
#         features[f'mfcc_{i}_kurtosis'] = pd.Series(mfccs[i]).kurtosis()

#     # Chroma features: main.py extracts 12 chroma means.
#     chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
#     for i in range(12): # Loop up to 12 to cover all chroma features
#         features[f'chroma_{i}_mean'] = np.mean(chroma[i])
    
#     # Spectral features (4 features)
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

#     # --- Prosodic Features ---
#     pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
#     valid_pitches = pitches[pitches > 0]
#     pitch_mean = np.mean(valid_pitches) if len(valid_pitches) > 0 else 0
    
#     features.update({
#         "pitch_mean": pitch_mean,
#         "pitch_std": np.std(valid_pitches) if len(valid_pitches) > 0 else 0,
#         "harmonic_ratio": np.mean(librosa.effects.harmonic(audio)),
#         "percussive_ratio": np.mean(librosa.effects.percussive(audio)),
#     })

#     # Voice activity features (2 features)
#     features.update(voice_activity_features(audio, sr))
    
#     # Jitter/Shimmer features (1 feature)
#     features['jitter'] = calculate_jitter(audio, sr) # Use the helper function

#     # Speaking rate and pauses (3 features)
#     intervals = librosa.effects.split(audio, top_db=25)
#     speech_duration = sum(end-start for start,end in intervals)/sr
#     total_duration = len(audio)/sr
#     features.update({
#         "pause_ratio": (total_duration - speech_duration) / total_duration if total_duration > 0 else 0,
#         "speaking_rate": len(intervals) / total_duration if total_duration > 0 else 0,
#         "speech_rate_var": np.std([(end-start)/sr for start,end in intervals]) if len(intervals) > 1 else 0,
#     })

#     # --- Linguistic features (if applicable) ---
#     # If your SELECTED_FEATURES list includes linguistic features (e.g., sentiment_polarity),
#     # you MUST uncomment and adapt this section to extract them from a transcript file.
#     # Otherwise, your feature count will be off, or the model will receive zeros for expected features.
#     # For simplicity, this example assumes your SELECTED_FEATURES do NOT include linguistic features,
#     # or that they were handled by the model being trained without them.

#     # Convert to DataFrame
#     # This creates a DataFrame with all computed features.
#     all_computed_features_df = pd.DataFrame([features])
    
#     # Select and reorder features based on SELECTED_FEATURES.
#     # This is crucial to ensure the DataFrame has the exact columns
#     # in the exact order that your trained model expects.
#     # fill_value=0 will fill any missing features (if SELECTED_FEATURES
#     # expects something not computed here) with 0.
#     final_features_df = all_computed_features_df.reindex(columns=SELECTED_FEATURES, fill_value=0)
    
#     # Verify the final feature count
#     if final_features_df.shape[1] != len(SELECTED_FEATURES):
#         print(f"Warning: Final feature count ({final_features_df.shape[1]}) does not match expected ({len(SELECTED_FEATURES)}).")
#         # You might want to raise an error or handle this more robustly
#         return None

#     print(f"Features extracted and aligned for prediction: {final_features_df.shape[1]}")
#     return final_features_df

# def predict_audio(file_path, pipeline):
#     """Make prediction using full pipeline"""
#     if not os.path.exists(file_path):
#         print(f"× File not found: {file_path}")
#         return "Error: File not found", 0.0
    
#     features_df = extract_features(file_path)
#     if features_df is None:
#         return "Error: Feature extraction failed", 0.0
    
#     try:
#         # Debug: Uncomment to verify feature count
#         # print(f"Feature shape before prediction: {features_df.shape}")
        
#         # Convert DataFrame to numpy array for prediction
#         pred = pipeline.predict(features_df.values)
#         proba = pipeline.predict_proba(features_df.values)
        
#         label = "Depressed" if pred[0] == 1 else "Not Depressed"
#         confidence = proba[0][pred[0]] * 100
#         return label, round(confidence, 1)
    
#     except Exception as e:
#         print(f"Prediction failed: {str(e)}")
#         # Print the shape of the features_df to help diagnose
#         print(f"Shape of features_df passed to pipeline: {features_df.shape}")
#         return "Error: Prediction failed", 0.0

# if __name__ == "__main__":
#     pipeline = load_model_pipeline()
#     if pipeline is None:
#         exit()
    
#     audio_file = "./Test/491_AUDIO.wav"
#     # Ensure the 'Test' directory exists and contains '491_AUDIO.wav'.
#     # If your model used linguistic features, you'll also need the corresponding
#     # transcript file (e.g., '491_TRANSCRIPT.csv') in the correct path.
    
#     label, confidence = predict_audio(audio_file, pipeline)
    
#     print("\n" + "="*50)
#     print(f"  Result: {label} ({confidence}% confidence)")
#     print("="*50)

import librosa
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler
# from textblob import TextBlob # Uncomment if your SELECTED_FEATURES include linguistic features and you have transcripts
import lightgbm as lgb # Added for LightGBM model loading

# This will be loaded dynamically from the saved feature names file
LOADED_FEATURE_NAMES = []

def load_model_pipeline():
    """Load the complete trained pipeline and its associated feature names"""
    try:
        # Change this path to load the specific model you want to test:
        # e.g., 'svm_model.joblib', 'random_forest_model.joblib', 'lightgbm_model.joblib', or 'ensemble_model.joblib'
        model_to_load = 'lightgbm_model.joblib' 
        pipeline = joblib.load(os.path.join('./trained_models', model_to_load))
        print(f"✔ Full model pipeline '{model_to_load}' loaded successfully")
        
        # Load the corresponding feature names.
        # If loading 'ensemble_model.joblib', it's best to use 'overall_selected_features.joblib'
        # as the ensemble itself doesn't have a single set of feature names.
        if model_to_load == 'lightgbm_model.joblib':
            feature_names_file = './trained_models/lightgbm_features.joblib'
        else:
            feature_names_file = os.path.join('./trained_models', model_to_load.replace('_model.joblib', '_features.joblib'))
            
        global LOADED_FEATURE_NAMES
        LOADED_FEATURE_NAMES = joblib.load(feature_names_file)

        print(f"✔ Expected features (from training): {len(LOADED_FEATURE_NAMES)} features")
        return pipeline
    except Exception as e:
        print(f"× Error loading model pipeline or feature names: {str(e)}")
        return None

def process_audio(audio_path, target_sr, duration):
    """
    Processes audio by loading, trimming/padding, silence removal,
    normalization, and pre-emphasis. Matches logic from main.py.
    """
    try:
        audio, sr = librosa.load(audio_path, sr=target_sr)

        # Trim or pad audio to fixed duration
        # Ensure integer for slicing/padding calculation
        if len(audio) < target_sr * duration:
            audio = np.pad(audio, (0, max(0, int(target_sr * duration) - len(audio))), 'constant')
        else:
            audio = audio[:int(target_sr * duration)]

        # Remove silence (less aggressive than before)
        audio, _ = librosa.effects.trim(audio, top_db=25)

        # Check for sufficient audio after trimming
        if len(audio) < sr * 0.5: # Require at least 0.5 seconds of audio
            print(f"Audio too short after trimming: {audio_path}")
            return None, None
        
        # Enhanced normalization and pre-emphasis
        audio = librosa.effects.preemphasis(audio)
        audio = librosa.util.normalize(audio)
        
        return audio, sr

    except Exception as e:
        print(f"An error occurred during audio processing for {audio_path}: {e}")
        return None, None


def voice_activity_features(audio, sr):
    """
    Calculate voice activity detection features.
    Copied directly from main.py to ensure consistency.
    """
    rms = librosa.feature.rms(y=audio)
    vad = rms > np.percentile(rms, 30)
    return {
        'vad_ratio': np.mean(vad),
        'vad_transitions': np.sum(np.diff(vad.astype(int)) != 0) 
    }

def calculate_jitter(audio, sr):
    """
    Helper function for jitter calculation.
    Copied from main.py's jitter_shimmer logic.
    """
    pitches = librosa.yin(audio, fmin=80, fmax=400)
    valid_pitches = pitches[pitches > 0]
    if len(valid_pitches) > 1:
        return np.mean(np.abs(np.diff(valid_pitches))) / np.mean(valid_pitches)
    return 0

def extract_features(file_path):
    """
    Extracts all *potential* features that were generated in main.py's
    extract_features function, then selects only the features specified
    in LOADED_FEATURE_NAMES.
    """
    features = {}
    
    # Process audio (matches the first part of extract_features in main.py)
    audio, sr = process_audio(file_path, target_sr=16000, duration=5)
    if audio is None:
        return None

    # --- Acoustic Features ---
    # MFCCs: main.py uses n_mfcc=40 and extracts mean, std, kurtosis for first 20.
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40) 
    for i in range(20): 
        features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
        features[f'mfcc_{i}_std'] = np.std(mfccs[i])
        features[f'mfcc_{i}_kurtosis'] = pd.Series(mfccs[i]).kurtosis()

    # Chroma features: main.py extracts 12 chroma means.
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    for i in range(12): 
        features[f'chroma_{i}_mean'] = np.mean(chroma[i])
    
    # Spectral features (4 features)
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

    # --- Prosodic Features ---
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    valid_pitches = pitches[pitches > 0]
    pitch_mean = np.mean(valid_pitches) if len(valid_pitches) > 0 else 0
    
    features.update({
        "pitch_mean": pitch_mean,
        "pitch_std": np.std(valid_pitches) if len(valid_pitches) > 0 else 0,
        "harmonic_ratio": np.mean(librosa.effects.harmonic(audio)),
        "percussive_ratio": np.mean(librosa.effects.percussive(audio)),
    })

    # Voice activity features (2 features)
    features.update(voice_activity_features(audio, sr))
    
    # Jitter/Shimmer features (1 feature)
    features['jitter'] = calculate_jitter(audio, sr) 

    # Speaking rate and pauses (3 features)
    intervals = librosa.effects.split(audio, top_db=25)
    speech_duration = sum(end-start for start,end in intervals)/sr
    total_duration = len(audio)/sr
    features.update({
        "pause_ratio": (total_duration - speech_duration) / total_duration if total_duration > 0 else 0,
        "speaking_rate": len(intervals) / total_duration if total_duration > 0 else 0,
        "speech_rate_var": np.std([(end-start)/sr for start,end in intervals]) if len(intervals) > 1 else 0,
    })

    # --- Linguistic features (if applicable) ---
    # If your LOADED_FEATURE_NAMES list includes linguistic features (e.g., sentiment_polarity),
    # you MUST uncomment and adapt this section to extract them from a transcript file.
    # Otherwise, your feature count will be off, or the model will receive zeros for expected features.
    # For simplicity, this example assumes your LOADED_FEATURE_NAMES do NOT include linguistic features,
    # or that they were handled by the model being trained without them.

    # Convert to DataFrame
    all_computed_features_df = pd.DataFrame([features])
    
    # Select and reorder features based on LOADED_FEATURE_NAMES.
    if not LOADED_FEATURE_NAMES:
        print("Error: Feature names not loaded. Call load_model_pipeline() first.")
        return None

    final_features_df = all_computed_features_df.reindex(columns=LOADED_FEATURE_NAMES, fill_value=0)
    
    # Verify the final feature count
    if final_features_df.shape[1] != len(LOADED_FEATURE_NAMES):
        print(f"Warning: Final feature count ({final_features_df.shape[1]}) does not match expected ({len(LOADED_FEATURE_NAMES)}).")
        return None

    print(f"Features extracted and aligned for prediction: {final_features_df.shape[1]}")
    return final_features_df

def predict_audio(file_path, pipeline):
    """Make prediction using full pipeline"""
    if not os.path.exists(file_path):
        print(f"× File not found: {file_path}")
        return "Error: File not found", 0.0
    
    features_df = extract_features(file_path)
    if features_df is None:
        return "Error: Feature extraction failed", 0.0
    
    try:
        pred = pipeline.predict(features_df.values)
        proba = pipeline.predict_proba(features_df.values)
        
        label = "Depressed" if pred[0] == 1 else "Not Depressed"
        confidence = proba[0][pred[0]] * 100
        return label, round(confidence, 1)
    
    except Exception as e:
        print(f"Prediction failed: {str(e)}")
        print(f"Shape of features_df passed to pipeline: {features_df.shape}")
        return "Error: Prediction failed", 0.0

if __name__ == "__main__":
    pipeline = load_model_pipeline()
    if pipeline is None:
        exit()
    
    audio_file = "./Test/309_AUDIO.wav"
    label, confidence = predict_audio(audio_file, pipeline)
    
    print("\n" + "="*50)
    print(f"  Result: {label} ({confidence}% confidence)")
    print("="*50)
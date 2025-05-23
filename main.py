import os
import glob
import pandas as pd
import librosa
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, roc_auc_score, average_precision_score)
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
from joblib import dump, load
import xgboost as xgb
from sklearn.ensemble import VotingClassifier

# ========================
# 1. CONFIGURATION
# ========================
DATASET_PATH = "Dataset"
OUTPUT_CSV = "all_features.csv"
MODEL_SAVE_PATH = "trained_models"

# Add PHQ-8 scores here (format: "participant_id": score)
PHQ8_SCORES = {
    "300": 2, "301": 3, "302": 4, "303": 0, "304": 6,
    "305": 7, "306": 0, "307": 4, "308": 22, "309": 15,
    "311": 21, "332": 18, "337": 10,     
}

# Create directory for saving models
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# ========================
# 2. DATA PREPROCESSING
# ========================
def process_audio(audio_path, target_sr, duration):
    try:
        # Load audio with target sample rate
        audio, sr = librosa.load(audio_path, sr=target_sr)

        # Trim or pad audio to fixed duration
        if len(audio) < target_sr * duration:
            audio = np.pad(audio, (0, max(0, target_sr * duration - len(audio))), 'constant')
        else:
            audio = audio[:int(target_sr * duration)] # Ensure integer for slicing

        # Remove silence (less aggressive than before)
        audio, _ = librosa.effects.trim(audio, top_db=25)

        # Check for sufficient audio after trimming
        if len(audio) < sr * 0.5: # Require at least 0.5 seconds of audio
            print(f"Audio too short after trimming: {audio_path}")
            return None

        # Enhanced normalization and pre-emphasis
        audio = librosa.effects.preemphasis(audio)
        audio = librosa.util.normalize(audio)

        return audio # Assuming you want to return the processed audio

    except Exception as e:
        print(f"An error occurred during audio processing for {audio_path}: {e}")
        return None # Or handle the error as appropriate for your application

# ========================
# 3. FEATURE EXTRACTION (ENHANCED)
# ========================
def voice_activity_features(audio, sr):
    """Calculate voice activity detection features"""
    rms = librosa.feature.rms(y=audio)
    vad = rms > np.percentile(rms, 30)
    return {
        'vad_ratio': np.mean(vad),
        'vad_transitions': np.sum(np.diff(vad.astype(int)) != 0)
    }

def jitter_shimmer(audio, sr):
    """Calculate voice quality metrics"""
    pitches = librosa.yin(audio, fmin=80, fmax=400)
    valid_pitches = pitches[pitches > 0]
    if len(valid_pitches) > 1:
        jitter = np.mean(np.abs(np.diff(valid_pitches))) / np.mean(valid_pitches)
        return {'jitter': jitter}
    return {'jitter': 0}

def extract_features(audio_path):
    """Enhanced feature extraction with new voice quality metrics"""
    base_name = os.path.basename(audio_path).split("_")[0]
    features = {"participant_id": base_name}

    # Load and preprocess audio
    audio = process_audio(audio_path, target_sr=16000, duration=5)  # or whatever sample rate/durati
    sr = 16000
    if audio is None:
        return None

    # 1. Acoustic features
    # MFCCs (with more detailed statistics)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    for i in range(20):  # Reduced from 40 to avoid overfitting
        features.update({
            f"mfcc_{i}_mean": np.mean(mfccs[i]),
            f"mfcc_{i}_std": np.std(mfccs[i]),
            f"mfcc_{i}_kurtosis": pd.Series(mfccs[i]).kurtosis(),
        })

    # Chroma features
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    features.update({
        f"chroma_{i}_mean": np.mean(chroma[i]) for i in range(12)
    })

    # Spectral features
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

    # 2. Prosodic features (enhanced)
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    valid_pitches = pitches[pitches > 0]
    pitch_mean = np.mean(valid_pitches) if len(valid_pitches) > 0 else 0
    
    features.update({
        "pitch_mean": pitch_mean,
        "pitch_std": np.std(valid_pitches) if len(valid_pitches) > 0 else 0,
        "harmonic_ratio": np.mean(librosa.effects.harmonic(audio)),
        "percussive_ratio": np.mean(librosa.effects.percussive(audio)),
    })

    # Add new voice quality features
    features.update(voice_activity_features(audio, sr))
    features.update(jitter_shimmer(audio, sr))

    # Speaking rate and pauses (more robust calculation)
    intervals = librosa.effects.split(audio, top_db=25)
    speech_duration = sum(end-start for start,end in intervals)/sr
    total_duration = len(audio)/sr
    features.update({
        "pause_ratio": (total_duration - speech_duration) / total_duration if total_duration > 0 else 0,
        "speaking_rate": len(intervals) / total_duration if total_duration > 0 else 0,
        "speech_rate_var": np.std([(end-start)/sr for start,end in intervals]) if len(intervals) > 1 else 0,
    })

    # 3. Linguistic features (from transcript)
    transcript_path = os.path.join(DATASET_PATH, f"{base_name}_TRANSCRIPT.csv")
    if os.path.exists(transcript_path):
        try:
            transcript = pd.read_csv(transcript_path, sep='\t')
            text = " ".join(str(x) for x in transcript['value'].dropna())
            blob = TextBlob(text)
            features.update({
                "sentiment_polarity": blob.sentiment.polarity,
                "sentiment_subjectivity": blob.sentiment.subjectivity,
                "word_count": len(text.split()),
                "avg_word_length": np.mean([len(word) for word in text.split()]) if text else 0,
                "lexical_diversity": len(set(text.split())) / len(text.split()) if text else 0,
                "negative_word_ratio": sum(1 for word in text.split() if blob.words[0].sentiment.polarity < -0.1) / len(text.split()) if text else 0
            })
        except Exception as e:
            print(f"Transcript error for {base_name}: {str(e)}")

    return features

# ========================
# 4. PROCESS ALL FILES
# ========================
def process_all_files():
    all_features = []
    audio_files = glob.glob(os.path.join(DATASET_PATH, "*_AUDIO.wav"))

    for audio_path in audio_files:
        print(f"Processing {os.path.basename(audio_path)}...")
        features = extract_features(audio_path)
        if features is not None:
            all_features.append(features)

    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)

    # Add labels if available
    if PHQ8_SCORES:
        features_df['phq8_score'] = features_df['participant_id'].map(PHQ8_SCORES)
        features_df['depression_label'] = (features_df['phq8_score'] >= 10).astype(int)

    # Save to CSV
    features_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved features for {len(all_features)} participants to {OUTPUT_CSV}")
    return features_df

# ========================
# 5. FEATURE SELECTION & PREPARATION
# ========================
def prepare_features(features_df):
    # Separate features and labels
    X = features_df.drop(columns=['participant_id', 'phq8_score', 'depression_label'], errors='ignore')
    y = features_df.get('depression_label', None)

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Feature selection (only if we have enough samples and labels)
    if y is not None and X_scaled.shape[0] > 1:
        selector = SelectKBest(f_classif, k=min(30, X_scaled.shape[1]))
        X_selected = selector.fit_transform(X_scaled, y)
        selected_features = X.columns[selector.get_support()]
        print(f"Selected top features: {selected_features.tolist()}")
    else:
        X_selected = X_scaled

    return X_selected, y, X.columns if y is None else selected_features

# ========================
# 6. MODEL TRAINING (ENHANCED)
# ========================
def train_models(X, y, feature_names):
    if y is None:
        print("No labels available for training")
        return None
    
    # Check class distribution
    class_counts = pd.Series(y).value_counts()
    print("\nClass distribution:")
    print(class_counts)
    
    if len(class_counts) < 2:
        print("\nWarning: Only one class present in data. Cannot train meaningful model.")
        return None
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    
    # Define models with balanced class weights
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight=class_weight_dict,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        ),
        "SVM": make_pipeline(
            StandardScaler(),
            SelectKBest(f_classif, k=12),
            CalibratedClassifierCV(
                SVC(
                    kernel='rbf',
                    class_weight=class_weight_dict,
                    probability=True,
                    gamma='scale',
                    C=1.0,
                    random_state=42
                ),
                method='isotonic',
                cv=3
            )
        ),
        "XGBoost": xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            scale_pos_weight=sum(y_train==0)/sum(y_train==1),
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
    }
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        try:
            # Train model with cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            print(f"Cross-validation ROC AUC: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}")
            
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else [0]*len(X_test)
            
            # Store results
            results[name] = {
                "model": model,
                "accuracy": accuracy_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else float('nan'),
                "avg_precision": average_precision_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else float('nan'),
                "report": classification_report(y_test, y_pred),
                "confusion_matrix": confusion_matrix(y_test, y_pred),
                "feature_importances": get_feature_importances(model, feature_names) if hasattr(model, 'feature_importances_') else None
            }
            
            # Print results
            print(f"\n{name} Results:")
            print(f"Accuracy: {results[name]['accuracy']:.2f}")
            print(f"ROC AUC: {results[name]['roc_auc']:.2f}")
            print("Classification Report:")
            print(results[name]['report'])
            
            # Plot confusion matrix
            plt.figure()
            sns.heatmap(results[name]['confusion_matrix'], 
                       annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Not Depressed', 'Depressed'],
                       yticklabels=['Not Depressed', 'Depressed'])
            plt.title(f"{name} Confusion Matrix")
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.show()
            
            # Save the trained model (with explicit error handling)
            model_filename = os.path.join(MODEL_SAVE_PATH, f"{name.lower().replace(' ', '_')}_model.joblib")
            try:
                dump(model, model_filename)
                print(f"✅ Saved {name} model to {model_filename}")
            except Exception as e:
                print(f"❌ Failed to save {name} model: {e}")
                continue
            
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            continue
    
    # Create and save an ensemble model if we have multiple successful models
    successful_models = [(name, res['model']) for name, res in results.items() if 'model' in res]
    if len(successful_models) > 1:
        ensemble = VotingClassifier(
            estimators=successful_models,
            voting='soft',
            flatten_transform=True
        )
        ensemble.fit(X_train, y_train)
        ensemble_filename = os.path.join(MODEL_SAVE_PATH, "ensemble_model.joblib")
        try:
            dump(ensemble, ensemble_filename)
            print(f"✅ Saved ensemble model to {ensemble_filename}")
            results['Ensemble'] = {
                "model": ensemble,
                "accuracy": accuracy_score(y_test, ensemble.predict(X_test))
            }
        except Exception as e:
            print(f"❌ Failed to save ensemble model: {e}")
    
    return results

def get_feature_importances(model, feature_names):
    """Extract feature importances in a readable format"""
    if hasattr(model, 'feature_importances_'):
        return sorted(zip(feature_names, model.feature_importances_), 
                     key=lambda x: x[1], reverse=True)
    elif hasattr(model, 'named_steps') and hasattr(model.named_steps['svc'], 'coef_'):
        # For SVM with feature selection
        coef = model.named_steps['svc'].coef_[0]
        selector = model.named_steps['selectkbest']
        selected_mask = selector.get_support()
        selected_features = feature_names[selected_mask]
        return sorted(zip(selected_features, np.abs(coef)), 
                     key=lambda x: x[1], reverse=True)
    return None

# ========================
# 7. VISUALIZATION (ENHANCED)
# ========================
def visualize_features(features_df):
    if 'depression_label' not in features_df.columns:
        print("No labels available for visualization")
        return

    plt.figure(figsize=(20, 15))
    
    # 1. Voice Quality Features
    plt.subplot(2, 2, 1)
    if 'jitter' in features_df.columns:
        sns.boxplot(data=features_df, x='depression_label', y='jitter')
        plt.title("Jitter by Depression Status")
    
    # 2. Prosodic Features
    plt.subplot(2, 2, 2)
    if 'pause_ratio' in features_df.columns and 'speaking_rate' in features_df.columns:
        sns.scatterplot(data=features_df, x='pause_ratio', y='speaking_rate', 
                        hue='depression_label', style='depression_label')
        plt.title("Pause Ratio vs Speaking Rate")
    
    # 3. MFCC Features
    plt.subplot(2, 2, 3)
    mfcc_means = [col for col in features_df.columns if col.startswith('mfcc_') and '_mean' in col][:5]
    if mfcc_means:
        mfcc_data = pd.melt(features_df[['depression_label'] + mfcc_means],
                            id_vars='depression_label')
        sns.boxplot(data=mfcc_data, x='variable', y='value', hue='depression_label')
        plt.title("MFCC Means Distribution")
        plt.xticks(rotation=45)
    
    # 4. Feature Correlation (only include available features)
    plt.subplot(2, 2, 4)
    possible_features = ['spectral_centroid_mean', 'pitch_mean', 'jitter',
                        'pause_ratio', 'sentiment_polarity', 'depression_label']
    available_features = [f for f in possible_features if f in features_df.columns]
    
    if len(available_features) > 1:  # Need at least 2 features for correlation
        sns.heatmap(features_df[available_features].corr(), annot=True, cmap='coolwarm')
        plt.title("Feature Correlation Matrix")
    else:
        plt.text(0.5, 0.5, "Not enough features for correlation matrix", 
                ha='center', va='center')
        plt.title("Feature Correlation (Not Available)")
    
    plt.tight_layout()
    plt.savefig("feature_analysis.png")
    plt.show()

# ========================
# 8. MAIN EXECUTION
# ========================
if __name__ == "__main__":
    # Step 1: Process all files and extract features
    print("Processing files and extracting features...")
    features_df = process_all_files()
    
    # Step 2: Visualize features
    print("\nVisualizing features...")
    visualize_features(features_df)
    
    # Step 3: Prepare features for ML
    print("\nPreparing features for machine learning...")
    X, y, feature_names = prepare_features(features_df)
    
    # Step 4: Train models (if labels available)
    if y is not None:
        print("\nTraining machine learning models...")
        results = train_models(X, y, feature_names)
        
        # Print feature importances
        for name, res in results.items():
            if res.get('feature_importances'):
                print(f"\n{name} Top Features:")
                for feat, imp in res['feature_importances'][:10]:
                    print(f"{feat}: {imp:.4f}")
    else:
        print("\nNo labels available. Skipping model training.")
        print("Please add PHQ-8 scores to the PHQ8_SCORES dictionary to enable model training.")
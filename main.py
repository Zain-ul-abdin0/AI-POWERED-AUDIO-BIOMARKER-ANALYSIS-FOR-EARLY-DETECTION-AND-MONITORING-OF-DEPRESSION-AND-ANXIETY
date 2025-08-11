import os
import glob
import pandas as pd
import librosa
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    ConfusionMatrixDisplay
)
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
from joblib import dump, load
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import VotingClassifier, StackingClassifier
import joblib

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# ========================
# 1. CONFIGURATION
# ========================
DATASET_PATH = "Dataset"
OUTPUT_CSV = "all_features.csv"
MODEL_SAVE_PATH = "trained_models"
TRANSCRIPTS_AVAILABLE = None

# PHQ-8 scores for depression classification (0-24 range)
# A score of >= 10 is generally considered a strong indicator of depression.
# '1' will now represent a score >= 10, and '0' will represent a score < 10.
PHQ8_SCORES = {
    "300": 2, "301": 3, "302": 4, "303": 0, "304": 6, "305": 7, "306": 0, "307": 4, "308": 22,
    "310": 4, "311": 21, "313": 7, "316": 6, "318": 3, "319": 13, "320": 11, "321": 20, "325": 10,
    "326": 2, "327": 4, "328": 4, "329": 1, "330": 12, "331": 8, "332": 18, "333": 5, "335": 12,
    "336": 7, "349": 5, "337": 10, "338": 15, "362": 20, "372": 13, "412": 12, "414": 16, "339": 11,
    "340": 1, "341": 7, "343": 9, "344": 11, "345": 15, "347": 16, "346": 23, "348": 20, "350": 11,
    "351": 14, "367": 19, "377": 16, "381": 16, "382": 0, "388": 17, "389": 14, "390": 9, "395": 7,
    "403": 0, "404": 0, "406": 2, "413": 10, "417": 7, "418": 10, "420": 3, "422": 12, "436": 0,
    "439": 1, "440": 19, "451": 4, "458": 5, "472": 3, "476": 3, "477": 2, "483": 15, "484": 9,
    "489": 3, "490": 2, "492": 0, "312": 2, "315": 2, "317": 8, "322": 5, "324": 5, "352": 10,
    "353": 11, "355": 10, "356": 10, "357": 7, "358": 7, "360": 4, "363": 0, "364": 0, "366": 0,
    "368": 7, "369": 0, "370": 0, "376": 12, "380": 10, "386": 11, "402": 11, "409": 10, "421": 10,
    "426": 20, "433": 10, "441": 18, "448": 18, "459": 16, "354": 18, "359": 13, "365": 12, "384": 15,
    "405": 17, "410": 12, "453": 17, "461": 17,
}

# Create directory for saving models
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# ========================
# 2. DATA PREPROCESSING
# ========================
def process_audio(audio_path, target_sr, duration):
    """
    Processes the audio file, including trimming/padding, silence removal,
    and normalization.
    """
    try:
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        
        if len(audio) < target_sr * duration:
            audio = np.pad(audio, (0, max(0, int(target_sr * duration) - len(audio))), 'constant')
        else:
            audio = audio[:int(target_sr * duration)]
            
        audio, _ = librosa.effects.trim(audio, top_db=25)
        
        if len(audio) < sr * 0.5:
            raise ValueError("Audio too short after trimming.")
            
        audio = librosa.effects.preemphasis(audio)
        audio = librosa.util.normalize(audio)
        
        return audio, sr
    except Exception as e:
        print(f"Audio processing error for {audio_path}: {e}")
        return None, None

def voice_activity_features(audio, sr):
    """Calculates voice activity detection features."""
    rms = librosa.feature.rms(y=audio)
    vad = rms > np.percentile(rms, 30)
    return {
        'vad_ratio': np.mean(vad),
        'vad_transitions': np.sum(np.diff(vad.astype(int)) != 0)
    }

def calculate_jitter(audio, sr):
    """Calculates jitter based on pitch changes."""
    pitches = librosa.yin(audio, fmin=80, fmax=400)
    valid_pitches = pitches[pitches > 0]
    if len(valid_pitches) > 1:
        return np.mean(np.abs(np.diff(valid_pitches))) / np.mean(valid_pitches)
    return 0
def detect_transcripts():
    """Check if any transcript files exist in the dataset."""
    transcript_files = glob.glob(os.path.join(DATASET_PATH, "*_TRANSCRIPT.csv"))
    return len(transcript_files) > 0
def extract_features(file_path, file_id):
    """Extracts a comprehensive set of audio features from a single file."""
    features = {}
    global TRANSCRIPTS_AVAILABLE
    
    try:
        audio, sr = process_audio(file_path, target_sr=16000, duration=5)
        if audio is None:
            return None

        # --- Acoustic Features ---
        n_mfcc_to_use = 20
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        for i in range(n_mfcc_to_use):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
            features[f'mfcc_{i}_kurtosis'] = pd.Series(mfccs[i]).kurtosis()

        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        for i in range(12):
            features[f'chroma_{i}_mean'] = np.mean(chroma[i])
        
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

        features.update(voice_activity_features(audio, sr))
        features['jitter'] = calculate_jitter(audio, sr)

        intervals = librosa.effects.split(audio, top_db=25)
        speech_duration = sum(end-start for start,end in intervals)/sr
        total_duration = len(audio)/sr
        features.update({
            "pause_ratio": (total_duration - speech_duration) / total_duration if total_duration > 0 else 0,
            "speaking_rate": len(intervals) / total_duration if total_duration > 0 else 0,
            "speech_rate_var": np.std([(end-start)/sr for start,end in intervals]) if len(intervals) > 1 else 0,
        })

        # --- Linguistic features (requires transcripts, which are not provided) ---
        # For a full implementation, you would need to parse transcripts here.
        if TRANSCRIPTS_AVAILABLE:
            transcript_path = os.path.join(DATASET_PATH, f"{file_id}_TRANSCRIPT.csv")
            print('Transcripts Available',transcript_path)
            print('Transcripts Variable Yes/No',TRANSCRIPTS_AVAILABLE)

            if os.path.exists(transcript_path):
                try:
                    transcript = pd.read_csv(transcript_path, sep="\t")
                    text = " ".join(str(x) for x in transcript["value"].dropna())
                    blob = TextBlob(text)
                    features.update({
                        "sentiment_polarity": blob.sentiment.polarity,
                        "sentiment_subjectivity": blob.sentiment.subjectivity,
                        "word_count": len(text.split()),
                        "avg_word_length": (
                            np.mean([len(word) for word in text.split()]) if text else 0
                        ),
                        "lexical_diversity": (
                            len(set(text.split())) / len(text.split()) if text else 0
                        ),
                        "negative_word_ratio": (
                            sum(
                                1
                                for word in text.split()
                                if TextBlob(word).sentiment.polarity < -0.1
                            ) / len(text.split())
                            if text
                            else 0
                        ),
                    })
                except Exception as e:
                    print(f"Transcript error for {file_id}: {str(e)}")
                    # Fill with neutral values if parsing fails
                    features.update(neutral_linguistic_features())
            else:
                # No transcript for this participant — fill with neutral values
                features.update(neutral_linguistic_features())

        # Add metadata for labeling
        features['participant_id'] = file_id
        features['phq8_score'] = PHQ8_SCORES.get(file_id)
        
        return features
        
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

def neutral_linguistic_features():
    """Return zero/neutral values for linguistic features."""
    return {
        "sentiment_polarity": 0.0,
        "sentiment_subjectivity": 0.0,
        "word_count": 0,
        "avg_word_length": 0.0,
        "lexical_diversity": 0.0,
        "negative_word_ratio": 0.0,
    }
def process_all_files():
    """Goes through the dataset, extracts features, and stores them in a DataFrame."""
    feature_list = []
    file_paths = glob.glob(os.path.join(DATASET_PATH, "**", "*.wav"), recursive=True)
    
    for file_path in file_paths:
        file_id = os.path.basename(file_path).split('_')[0]
        if file_id in PHQ8_SCORES:
            features = extract_features(file_path, file_id)
            if features:
                feature_list.append(features)
    
    features_df = pd.DataFrame(feature_list)
    features_df.to_csv(OUTPUT_CSV, index=False)
    return features_df

def visualize_features(df):
    """
    Creates visualizations of the extracted features to identify trends.
    This helps in feature selection and understanding the data.
    """
    # Create the binary label for depression based on PHQ-8 score >= 10
    df['depression_label'] = df['phq8_score'].apply(lambda x: 1 if x >= 10 else 0)
    
    print("\nVisualizing top 10 features by correlation...")
    
    # Calculate correlation matrix
    correlations = df.corr(numeric_only=True)['depression_label'].sort_values(ascending=False)
    top_10_correlated_features = correlations.index[1:11] 
    
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(top_10_correlated_features):
        plt.subplot(2, 5, i + 1)
        sns.boxplot(x='depression_label', y=feature, data=df)
        plt.title(f'Depression Label vs. {feature}', fontsize=8)
        plt.xticks([0, 1], ['Not Depressed', 'Depressed'])
    plt.tight_layout()
    plt.show()

def prepare_features(df):
    """
    Prepares the feature DataFrame for machine learning, including
    imputation, scaling, and feature selection.
    """
    # Create the binary label
    df['depression_label'] = df['phq8_score'].apply(lambda x: 1 if x >= 10 else 0)
    
    # Separate features and labels
    feature_columns = [col for col in df.columns if col not in ['participant_id', 'phq8_score', 'depression_label']]
    X = df[feature_columns]
    y = df['depression_label']
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=feature_columns)
    
    # Select top K best features using ANOVA F-value
    selector = SelectKBest(f_classif, k=30)
    X_selected = selector.fit_transform(X, y)
    selected_mask = selector.get_support()
    selected_features = X.columns[selected_mask]
    
    # Save the list of selected feature names for the API
    joblib.dump(selected_features.tolist(), os.path.join(MODEL_SAVE_PATH, 'overall_selected_features.joblib'))
    
    return X_selected, y, selected_features

def plot_confusion_matrix(y_true, y_pred, model_name):
    """
    Plots a confusion matrix heatmap for a given model.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Depressed', 'Depressed'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()

def train_models(X, y, feature_names):
    """
    Trains multiple models, including an improved ensemble, and
    saves all trained models and their feature importances.
    """
    results = {}
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Calculate class weights for imbalanced data
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

    # --- Model Pipelines with Hyperparameter Tuning ---
    
    # 1. LightGBM Classifier with RandomizedSearchCV
    lgbm = make_pipeline(StandardScaler(), lgb.LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1))
    lgbm_param_dist = {
        'lgbmclassifier__n_estimators': [100, 200, 300],
        'lgbmclassifier__learning_rate': [0.01, 0.05, 0.1],
        'lgbmclassifier__num_leaves': [20, 31, 40],
    }
    lgbm_gs = RandomizedSearchCV(lgbm, lgbm_param_dist, n_iter=10, cv=5, scoring='roc_auc', random_state=42)
    lgbm_gs.fit(X_train, y_train)
    results['LightGBM'] = {'model': lgbm_gs.best_estimator_, 'best_params': lgbm_gs.best_params_}
    joblib.dump(lgbm_gs.best_estimator_, os.path.join(MODEL_SAVE_PATH, 'lightgbm_model.joblib'))
    print(f"\n✔ LightGBM model saved to {os.path.join(MODEL_SAVE_PATH, 'lightgbm_model.joblib')}")
    
    # 2. Random Forest Classifier with RandomizedSearchCV
    rf = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42, class_weight='balanced'))
    rf_param_dist = {
        'randomforestclassifier__n_estimators': [100, 200, 300],
        'randomforestclassifier__max_depth': [None, 10, 20],
        'randomforestclassifier__min_samples_leaf': [1, 2, 4],
    }
    rf_gs = RandomizedSearchCV(rf, rf_param_dist, n_iter=10, cv=5, scoring='roc_auc', random_state=42)
    rf_gs.fit(X_train, y_train)
    results['RandomForest'] = {'model': rf_gs.best_estimator_, 'best_params': rf_gs.best_params_}
    joblib.dump(rf_gs.best_estimator_, os.path.join(MODEL_SAVE_PATH, 'random_forest_model.joblib'))
    print(f"✔ RandomForest model saved to {os.path.join(MODEL_SAVE_PATH, 'random_forest_model.joblib')}")
    
    # 3. XGBoost Classifier with RandomizedSearchCV
    xgb_model = make_pipeline(
        StandardScaler(), 
        xgb.XGBClassifier(
            random_state=42, 
            eval_metric='logloss',
            use_label_encoder=False,
            scale_pos_weight=class_weight_dict.get(1) / class_weight_dict.get(0) if class_weight_dict.get(0) else 1
        )
    )
    xgb_param_dist = {
        'xgbclassifier__n_estimators': [100, 200, 300],
        'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
        'xgbclassifier__max_depth': [3, 5, 7],
        'xgbclassifier__subsample': [0.8, 1.0],
    }
    xgb_gs = RandomizedSearchCV(xgb_model, xgb_param_dist, n_iter=10, cv=5, scoring='roc_auc', random_state=42)
    xgb_gs.fit(X_train, y_train)
    results['XGBoost'] = {'model': xgb_gs.best_estimator_, 'best_params': xgb_gs.best_params_}
    joblib.dump(xgb_gs.best_estimator_, os.path.join(MODEL_SAVE_PATH, 'xgboost_model.joblib'))
    print(f"✔ XGBoost model saved to {os.path.join(MODEL_SAVE_PATH, 'xgboost_model.joblib')}")

    # 4. Support Vector Machine (SVC) with probability calibration
    svm = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced'))
    svm_calibrated = CalibratedClassifierCV(svm, method='isotonic', cv=5)
    svm_calibrated.fit(X_train, y_train)
    results['SVC'] = {'model': svm_calibrated}
    joblib.dump(svm_calibrated, os.path.join(MODEL_SAVE_PATH, 'svm_model.joblib'))
    print(f"✔ SVC model saved to {os.path.join(MODEL_SAVE_PATH, 'svm_model.joblib')}")

    # --- Ensemble Model (StackingClassifier) ---
    estimators = [
        ('lgbm', lgbm_gs.best_estimator_),
        ('rf', rf_gs.best_estimator_),
        ('xgb', xgb_gs.best_estimator_),
        ('svm', svm_calibrated)
    ]
    ensemble = StackingClassifier(estimators=estimators, final_estimator=xgb.XGBClassifier(random_state=42), cv=5)
    ensemble.fit(X_train, y_train)
    results['Ensemble'] = {'model': ensemble}
    joblib.dump(ensemble, os.path.join(MODEL_SAVE_PATH, 'ensemble_model.joblib'))
    print(f"✔ Ensemble model saved to {os.path.join(MODEL_SAVE_PATH, 'ensemble_model.joblib')}")

    print("\nTraining and evaluation complete. Results:")
    
    # Evaluate and visualize all models
    for name, res in results.items():
        model = res['model']
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        print(f"\n--- {name} Results ---")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
        print(f"Avg Precision: {average_precision_score(y_test, y_proba):.4f}")
        print(classification_report(y_test, y_pred))
        
        # Plot and show the confusion matrix
        plot_confusion_matrix(y_test, y_pred, name)
        
        # Save feature importances for tree-based models
        if name in ['RandomForest', 'LightGBM', 'XGBoost']:
            final_model = None
            if hasattr(model, 'named_steps'):
                for step_name, step_model in model.named_steps.items():
                    if hasattr(step_model, 'predict'):
                        final_model = step_model
                        break
            else: # For calibrated models like SVM
                final_model = model
            
            if final_model and hasattr(final_model, 'feature_importances_'):
                importances = final_model.feature_importances_
                feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
                feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
                
                # Save feature importances to a joblib file
                importance_filename = f'{name.lower()}_feature_importances.joblib'
                joblib.dump(feature_importance_df.to_records(index=False).tolist(), os.path.join(MODEL_SAVE_PATH, importance_filename))
                print(f"✔ Feature importances for {name} saved to {importance_filename}")
                
                res['feature_importances'] = feature_importance_df.to_records(index=False).tolist()
            else:
                print(f"Could not get feature importances for model: {name}")

    return results

# ========================
# 8. MAIN EXECUTION
# =======================
if __name__ == "__main__":
    # Step 1: Process all files and extract features
    TRANSCRIPTS_AVAILABLE = detect_transcripts()
    print(f"Transcripts available: {TRANSCRIPTS_AVAILABLE}")
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
        
        selected_features = joblib.load("trained_models/overall_selected_features.joblib")
        print("\nSelected 30 Features:")
        for i, feat in enumerate(selected_features, 1):
            print(f"{i}. {feat}")

        # Print feature importances
        for name, res in results.items():
            if res.get("feature_importances"):
                print(f"\n{name} Top Features:")
                for feat, imp in res["feature_importances"][:10]:
                    print(f"{feat}: {imp:.4f}")
    else:
        print("No labels found, cannot train models.")

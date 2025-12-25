import os
import glob
import pandas as pd
import librosa
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import StackingClassifier
import joblib

import warnings
warnings.filterwarnings("ignore")

AUDIO_DATA_FOLDER = "Dataset"
FEATURES_OUTPUT_FILE = "all_features.csv"
MODELS_SAVE_FOLDER = "trained_models"
HAS_TRANSCRIPTS = None

PATIENT_DEPRESSION_SCORES = {
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

os.makedirs(MODELS_SAVE_FOLDER, exist_ok=True)
def prepare_audio(audio_file_path, target_sample_rate, clip_duration):
    try:
        audio_data, original_sample_rate = librosa.load(audio_file_path, sr=target_sample_rate, mono=True)
        
        if len(audio_data) < target_sample_rate * clip_duration:
            audio_data = np.pad(audio_data, (0, max(0, int(target_sample_rate * clip_duration) - len(audio_data))), 'constant')
        else:
            audio_data = audio_data[:int(target_sample_rate * clip_duration)]
            
        audio_data, _ = librosa.effects.trim(audio_data, top_db=25)
        
        if len(audio_data) < original_sample_rate * 0.5:
            raise ValueError("Audio too short after trimming.")
            
        audio_data = librosa.effects.preemphasis(audio_data)
        audio_data = librosa.util.normalize(audio_data)
        
        return audio_data, original_sample_rate
    except Exception as error:
        print(f"Could not process {audio_file_path}: {error}")
        return None, None

def detect_voice_activity(audio_data, sample_rate):
    vad_info = detect_voice_activity(audio_data, sample_rate)
    volume_levels = librosa.feature.rms(y=audio_data)
    voice_detected = volume_levels > np.percentile(volume_levels, 30)
    print(f"Patient  | VAD → "
      f"Speech Ratio: {vad_info['speech_ratio']:.2f}, "
      f"Speech Transitions: {vad_info['speech_transitions']}")
    return {
        'speech_ratio': np.mean(voice_detected),
        'speech_transitions': np.sum(np.diff(voice_detected.astype(int)) != 0)
    }

def measure_vocal_jitter(audio_data, sample_rate):
    pitch_values = librosa.yin(audio_data, fmin=80, fmax=400)
    valid_pitches = pitch_values[pitch_values > 0]
    if len(valid_pitches) > 1:
        return np.mean(np.abs(np.diff(valid_pitches))) / np.mean(valid_pitches)
    return 0

def check_for_transcripts():
    transcript_files = glob.glob(os.path.join(AUDIO_DATA_FOLDER, "*_TRANSCRIPT.csv"))
    return len(transcript_files) > 0

def extract_audio_features(audio_file_path, patient_id):
    feature_set = {}
    global HAS_TRANSCRIPTS
    
    try:
        audio_data, sample_rate = prepare_audio(audio_file_path, target_sample_rate=16000, clip_duration=5)
        if audio_data is None:
            return None
        mfcc_count = 20
        mfcc_features = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        for i in range(mfcc_count):
            feature_set[f'mfcc_{i}_mean'] = np.mean(mfcc_features[i])
            feature_set[f'mfcc_{i}_std'] = np.std(mfcc_features[i])
            feature_set[f'mfcc_{i}_kurtosis'] = pd.Series(mfcc_features[i]).kurtosis()

        chroma_features = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
        for i in range(12):
            feature_set[f'chroma_{i}_mean'] = np.mean(chroma_features[i])
        
        spectral_center = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)
        frequency_range = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate)
        feature_set.update({
            "spectral_centroid_mean": np.mean(spectral_center),
            "spectral_bandwidth_mean": np.mean(frequency_range),
            "spectral_rolloff_mean": np.mean(spectral_rolloff),
            "spectral_contrast_mean": np.mean(spectral_contrast),
        })

        pitch_values, strength_values = librosa.piptrack(y=audio_data, sr=sample_rate)
        valid_pitches = pitch_values[pitch_values > 0]
        average_pitch = np.mean(valid_pitches) if len(valid_pitches) > 0 else 0
        
        feature_set.update({
            "pitch_mean": average_pitch,
            "pitch_std": np.std(valid_pitches) if len(valid_pitches) > 0 else 0,
            "harmonic_ratio": np.mean(librosa.effects.harmonic(audio_data)),
            "percussive_ratio": np.mean(librosa.effects.percussive(audio_data)),
        })

        feature_set.update(detect_voice_activity(audio_data, sample_rate))
        feature_set['jitter'] = measure_vocal_jitter(audio_data, sample_rate)

        speech_segments = librosa.effects.split(audio_data, top_db=25)
        speaking_time = sum(end-start for start,end in speech_segments)/sample_rate
        total_time = len(audio_data)/sample_rate
        feature_set.update({
            "pause_ratio": (total_time - speaking_time) / total_time if total_time > 0 else 0,
            "speaking_rate": len(speech_segments) / total_time if total_time > 0 else 0,
            "speech_rate_var": np.std([(end-start)/sample_rate for start,end in speech_segments]) if len(speech_segments) > 1 else 0,
        })

        if HAS_TRANSCRIPTS:
            transcript_file_path = os.path.join(AUDIO_DATA_FOLDER, f"{patient_id}_TRANSCRIPT.csv")
            print('Transcripts Available', transcript_file_path)

            if os.path.exists(transcript_file_path):
                try:
                    transcript_data = pd.read_csv(transcript_file_path, sep="\t")
                    all_text = " ".join(str(x) for x in transcript_data["value"].dropna())
                    text_analysis = TextBlob(all_text)
                    feature_set.update({
                        "sentiment_polarity": text_analysis.sentiment.polarity,
                        "sentiment_subjectivity": text_analysis.sentiment.subjectivity,
                        "word_count": len(all_text.split()),
                        "avg_word_length": (
                            np.mean([len(word) for word in all_text.split()]) if all_text else 0
                        ),
                        "lexical_diversity": (
                            len(set(all_text.split())) / len(all_text.split()) if all_text else 0
                        ),
                        "negative_word_ratio": (
                            sum(
                                1
                                for word in all_text.split()
                                if TextBlob(word).sentiment.polarity < -0.1
                            ) / len(all_text.split())
                            if all_text
                            else 0
                        ),
                    })
                except Exception as error:
                    print(f"Could not read transcript for {patient_id}: {str(error)}")
                    feature_set.update(get_neutral_language_features())
            else:
                feature_set.update(get_neutral_language_features())

        feature_set['patient_id'] = patient_id
        feature_set['depression_score'] = PATIENT_DEPRESSION_SCORES.get(patient_id)
        
        return feature_set
        
    except Exception as error:
        print(f"Error analyzing {audio_file_path}: {error}")
        return None

def get_neutral_language_features():
    return {
        "sentiment_polarity": 0.0,
        "sentiment_subjectivity": 0.0,
        "word_count": 0,
        "avg_word_length": 0.0,
        "lexical_diversity": 0.0,
        "negative_word_ratio": 0.0,
    }

def process_all_audio_files():
    all_features = []
    audio_files = glob.glob(os.path.join(AUDIO_DATA_FOLDER, "**", "*.wav"), recursive=True)
    
    for audio_file in audio_files:
        patient_id = os.path.basename(audio_file).split('_')[0]
        if patient_id in PATIENT_DEPRESSION_SCORES:
            features = extract_audio_features(audio_file, patient_id)
            if features:
                all_features.append(features)
    
    features_dataframe = pd.DataFrame(all_features)
    features_dataframe.to_csv(FEATURES_OUTPUT_FILE, index=False)
    return features_dataframe

def create_feature_visualizations(dataframe):
    dataframe['has_depression'] = dataframe['depression_score'].apply(lambda x: 1 if x >= 10 else 0)    
    print("\nCreating visualizations for top 10 most relevant features...")
    correlations = dataframe.corr(numeric_only=True)['has_depression'].sort_values(ascending=False)
    top_10_features = correlations.index[1:11] 
    
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(top_10_features):
        plt.subplot(2, 5, i + 1)
        sns.boxplot(x='has_depression', y=feature, data=dataframe)
        plt.title(f'Depression vs. {feature}', fontsize=8)
        plt.xticks([0, 1], ['Not Depressed', 'Depressed'])
    plt.tight_layout()
    plt.show()

def prepare_data_for_training(dataframe):
    dataframe['has_depression'] = dataframe['depression_score'].apply(lambda x: 1 if x >= 10 else 0)
    
    feature_columns = [col for col in dataframe.columns if col not in ['patient_id', 'depression_score', 'has_depression']]
    features = dataframe[feature_columns]
    labels = dataframe['has_depression']
    
    missing_value_handler = SimpleImputer(strategy='mean')
    features = pd.DataFrame(missing_value_handler.fit_transform(features), columns=feature_columns)
    
    feature_selector = SelectKBest(f_classif, k=30)
    selected_features = feature_selector.fit_transform(features, labels)
    selected_mask = feature_selector.get_support()
    chosen_features = features.columns[selected_mask]
    
    joblib.dump(chosen_features.tolist(), os.path.join(MODELS_SAVE_FOLDER, 'selected_features.joblib'))
    
    return selected_features, labels, chosen_features

def show_confusion_matrix(true_labels, predicted_labels, model_name):
    matrix = confusion_matrix(true_labels, predicted_labels)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=['Not Depressed', 'Depressed'])
    display.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()

def train_machine_learning_models(features, labels, feature_names):
    results = {}
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)
    
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    lightgbm_model = make_pipeline(StandardScaler(), lgb.LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1))
    lightgbm_params = {
        'lgbmclassifier__n_estimators': [100, 200, 300],
        'lgbmclassifier__learning_rate': [0.01, 0.05, 0.1],
        'lgbmclassifier__num_leaves': [20, 31, 40],
    }
    lightgbm_search = RandomizedSearchCV(lightgbm_model, lightgbm_params, n_iter=10, cv=5, scoring='roc_auc', random_state=42)
    lightgbm_search.fit(X_train, y_train)
    results['LightGBM'] = {'model': lightgbm_search.best_estimator_, 'best_params': lightgbm_search.best_params_}
    joblib.dump(lightgbm_search.best_estimator_, os.path.join(MODELS_SAVE_FOLDER, 'lightgbm_model.joblib'))
    print(f"\n✔ LightGBM model saved to {os.path.join(MODELS_SAVE_FOLDER, 'lightgbm_model.joblib')}")
    
    random_forest_model = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42, class_weight='balanced'))
    random_forest_params = {
        'randomforestclassifier__n_estimators': [100, 200, 300],
        'randomforestclassifier__max_depth': [None, 10, 20],
        'randomforestclassifier__min_samples_leaf': [1, 2, 4],
    }
    random_forest_search = RandomizedSearchCV(random_forest_model, random_forest_params, n_iter=10, cv=5, scoring='roc_auc', random_state=42)
    random_forest_search.fit(X_train, y_train)
    results['RandomForest'] = {'model': random_forest_search.best_estimator_, 'best_params': random_forest_search.best_params_}
    joblib.dump(random_forest_search.best_estimator_, os.path.join(MODELS_SAVE_FOLDER, 'random_forest_model.joblib'))
    print(f"✔ RandomForest model saved to {os.path.join(MODELS_SAVE_FOLDER, 'random_forest_model.joblib')}")
    
    xgboost_model = make_pipeline(
        StandardScaler(), 
        xgb.XGBClassifier(
            random_state=42, 
            eval_metric='logloss',
            use_label_encoder=False,
            scale_pos_weight=weight_dict.get(1) / weight_dict.get(0) if weight_dict.get(0) else 1
        )
    )
    xgboost_params = {
        'xgbclassifier__n_estimators': [100, 200, 300],
        'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
        'xgbclassifier__max_depth': [3, 5, 7],
        'xgbclassifier__subsample': [0.8, 1.0],
    }
    xgboost_search = RandomizedSearchCV(xgboost_model, xgboost_params, n_iter=10, cv=5, scoring='roc_auc', random_state=42)
    xgboost_search.fit(X_train, y_train)
    results['XGBoost'] = {'model': xgboost_search.best_estimator_, 'best_params': xgboost_search.best_params_}
    joblib.dump(xgboost_search.best_estimator_, os.path.join(MODELS_SAVE_FOLDER, 'xgboost_model.joblib'))
    print(f"✔ XGBoost model saved to {os.path.join(MODELS_SAVE_FOLDER, 'xgboost_model.joblib')}")

    svm_model = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced'))
    calibrated_svm = CalibratedClassifierCV(svm_model, method='isotonic', cv=5)
    calibrated_svm.fit(X_train, y_train)
    results['SVC'] = {'model': calibrated_svm}
    joblib.dump(calibrated_svm, os.path.join(MODELS_SAVE_FOLDER, 'svm_model.joblib'))
    print(f"✔ SVC model saved to {os.path.join(MODELS_SAVE_FOLDER, 'svm_model.joblib')}")
    
    model_collection = [
        ('lgbm', lightgbm_search.best_estimator_),
        ('rf', random_forest_search.best_estimator_),
        ('xgb', xgboost_search.best_estimator_),
        ('svm', calibrated_svm)
    ]
    combined_model = StackingClassifier(estimators=model_collection, final_estimator=xgb.XGBClassifier(random_state=42), cv=5)
    combined_model.fit(X_train, y_train)
    results['Ensemble'] = {'model': combined_model}
    joblib.dump(combined_model, os.path.join(MODELS_SAVE_FOLDER, 'ensemble_model.joblib'))
    print(f"✔ Ensemble model saved to {os.path.join(MODELS_SAVE_FOLDER, 'ensemble_model.joblib')}")

    print("\nTraining complete. Evaluating model performance:")
    
    for model_name, model_info in results.items():
        current_model = model_info['model']
        predictions = current_model.predict(X_test)
        prediction_probabilities = current_model.predict_proba(X_test)[:, 1]
        
        print(f"\n--- {model_name} Performance ---")
        print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
        print(f"ROC AUC: {roc_auc_score(y_test, prediction_probabilities):.4f}")
        print(f"Avg Precision: {average_precision_score(y_test, prediction_probabilities):.4f}")
        print(classification_report(y_test, predictions))
        
        show_confusion_matrix(y_test, predictions, model_name)
        
        if model_name in ['RandomForest', 'LightGBM', 'XGBoost']:
            trained_model = None
            if hasattr(current_model, 'named_steps'):
                for step_name, step_model in current_model.named_steps.items():
                    if hasattr(step_model, 'predict'):
                        trained_model = step_model
                        break
            else:
                trained_model = current_model
            
            if trained_model and hasattr(trained_model, 'feature_importances_'):
                importance_scores = trained_model.feature_importances_
                importance_df = pd.DataFrame({'feature': feature_names, 'importance': importance_scores})
                importance_df = importance_df.sort_values(by='importance', ascending=False)
                
                # Save feature importance rankings
                importance_file = f'{model_name.lower()}_feature_importances.joblib'
                joblib.dump(importance_df.to_records(index=False).tolist(), os.path.join(MODELS_SAVE_FOLDER, importance_file))
                print(f"✔ Feature importance for {model_name} saved to {importance_file}")
                
                model_info['feature_importances'] = importance_df.to_records(index=False).tolist()
            else:
                print(f"Could not get feature importance for model: {model_name}")
    
    best_model = ''
    best_score = -1

    for model_name, model_info in results.items():
        current_model = model_info['model']
        prediction_probabilities = current_model.predict_proba(X_test)[:, 1]
        roc_score = roc_auc_score(y_test, prediction_probabilities)
        
        if roc_score > best_score:
            best_score = roc_score
            best_model = model_name

    print("\n=========================")
    print(f"🥇 Best performing model: {best_model}")
    print(f"🏆 ROC AUC score: {best_score:.4f}")
    print("=========================")

    return results

if __name__ == "__main__":
    print("Processing audio files and extracting features...")
    features_data = process_all_audio_files()

    print("\nCreating feature visualizations...")
    create_feature_visualizations(features_data)

    print("\nPreparing data for machine learning...")
    X_features, y_labels, selected_feature_names = prepare_data_for_training(features_data)
    
    if y_labels is not None:
        print("\nTraining machine learning models...")
        model_results = train_machine_learning_models(X_features, y_labels, selected_feature_names)
        
        selected_features = joblib.load("trained_models/selected_features.joblib")
        print("\nTop 30 Selected Features:")
        for i, feature in enumerate(selected_features, 1):
            print(f"{i}. {feature}")

        for model_name, model_info in model_results.items():
            if model_info.get("feature_importances"):
                print(f"\n{model_name} Most Important Features:")
                for feature_name, importance_score in model_info["feature_importances"][:10]:
                    print(f"{feature_name}: {importance_score:.4f}")
    else:
        print("No depression scores available, cannot train models.")
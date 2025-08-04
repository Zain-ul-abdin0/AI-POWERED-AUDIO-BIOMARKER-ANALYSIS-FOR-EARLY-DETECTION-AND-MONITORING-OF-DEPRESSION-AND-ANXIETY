# import os
# import glob
# import pandas as pd
# import librosa
# import numpy as np
# from textblob import TextBlob
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.metrics import (
#     classification_report,
#     confusion_matrix,
#     accuracy_score,
#     roc_auc_score,
#     average_precision_score,
# )
# from sklearn.pipeline import make_pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.decomposition import PCA
# from sklearn.feature_selection import SelectKBest, f_classif
# from sklearn.calibration import CalibratedClassifierCV
# from sklearn.utils.class_weight import compute_class_weight
# from joblib import dump, load
# import lightgbm as lgb  # Changed from xgboost
# import xgboost as xgb
# from sklearn.ensemble import VotingClassifier
# import joblib


# # ========================
# # 1. CONFIGURATION
# # ========================
# DATASET_PATH = "Dataset"
# OUTPUT_CSV = "all_features.csv"
# MODEL_SAVE_PATH = "trained_models"

# # data balance  Add PHQ-8 scores here (format: "participant_id": score)
# # PHQ8_SCORES = {
# # "302": 4,     "307": 4,     "331": 8,     "335": 12,     "346": 23,
# # "367": 19,     "377": 16,     "381": 16,     "382": 0,     "388": 17,
# # "389": 14,     "390": 9,     "395": 7,     "403": 0,     "404": 0,
# # "406": 2,     "413": 10,     "417": 7,     "418": 10,     "420": 3,
# # "422": 12,     "436": 0,     "439": 1,     "440": 19,     "451": 4,
# # "458": 5,     "472": 3,     "476": 3,     "477": 2,
# # # "482": 1,
# # "483": 15,     "484": 9,     "489": 3,     "490": 2,     "492": 0,
# # "303": 0,     "304": 6,     "305": 7,     "310": 4,     "312": 2,
# # "313": 7,     "315": 2,     "316": 6,     "317": 8,     "318": 3,
# # "319": 13,     "320": 11,     "321": 20,     "322": 5,     "324": 5,
# # "325": 10,     "326": 2,     "327": 4,     "328": 4,     "330": 12,
# # "333": 5,     "336": 7,     "338": 15,     "339": 11,     "340": 1,
# # "341": 7,     "343": 9,     "344": 11,     "345": 15,     "347": 16,
# # "348": 20,     "350": 11,     "351": 14,     "352": 10,     "353": 11,
# # "355": 10,     "356": 10,     "357": 7,     "358": 7,     "360": 4,
# # "362": 20,     "363": 0,     "364": 0,     "366": 0,     "368": 7,
# # "369": 0,     "370": 0,     "371": 9,     "372": 13,     "374": 2,
# # "375": 5,     "376": 12,     "379": 2,     "380": 10,     "383": 7,
# # "385": 8,     "386": 11,     "391": 9,     "392": 1,     "393": 2,
# # "397": 5,     "400": 7,     "401": 9,     "402": 11,     "409": 10,
# # "412": 12,     "414": 16,     "415": 3,     "416": 3,     "419": 3,
# # "423": 0,     "425": 6,     "426": 20,     "427": 5,     "428": 0,
# # "429": 1,     "430": 3,     "433": 10,     "434": 2,     "437": 0,
# # "441": 18,     "443": 1,     "444": 7,     "445": 1,     "446": 0,
# # "447": 1,     "448": 18,     "449": 2,     "454": 1,     "455": 1,
# # "456": 6,     "457": 3,     "459": 16,     "463": 0,     "464": 0,
# # "468": 4,     "471": 0,     "473": 0,     "474": 4,     "475": 6,
# # "478": 1,     "479": 7,     "485": 2,     "486": 4,     "487": 0,
# # "488": 0,
# # #   "491": 8,
# # "300": 2,     "301": 3,     "306": 0,
# # # "308": 22,

# # # "309": 15,

# # "311": 21,     "314": 1,     "323": 1,
# # "329": 1,     "332": 18,     "334": 5,     "337": 10,     "349": 5,
# # "354": 18,     "359": 13,     "361": 0,     "365": 12,     "373": 9,
# # "378": 1,     "384": 15,     "387": 2,     "396": 5,     "399": 7,
# # "405": 17,     "407": 3,     "408": 0,     "410": 12,     "411": 0,
# # "421": 10,     "424": 3,     "431": 2,     "432": 1,     "435": 8,
# # "438": 2,     "442": 6,     "450": 9,     "452": 1,     "453": 17,
# # "461": 17,     "462": 9,     "465": 2,     "466": 9,     "467": 0,
# # "469": 3,     "470": 3,     "480": 1,     "481": 7 }

# # Data Balance
# PHQ8_SCORES = {
#     "300": 2,
#     "301": 3,
#     "302": 4,
#     "303": 0,
#     "304": 6,
#     "305": 7,
#     "306": 0,
#     "307": 4,
#     "308": 22,
#     "310": 4,
#     "311": 21,
#     "313": 7,
#     "316": 6,
#     "318": 3,
#     "319": 13,
#     "320": 11,
#     "321": 20,
#     "325": 10,
#     "326": 2,
#     "327": 4,
#     "328": 4,
#     "329": 1,
#     "330": 12,
#     "331": 8,
#     "332": 18,
#     "333": 5,
#     "335": 12,
#     "336": 7,
#     "349": 5,
#     "337": 10,
#     "338": 15,
#     "362": 20,
#     "372": 13,
#     "412": 12,
#     "414": 16,
#     "339": 11,
#     "340": 1,
#     "341": 7,
#     "343": 9,
#     "344": 11,
#     "345": 15,
#     "347": 16,
#     "346": 23,
#     "348": 20,
#     "350": 11,
#     "351": 14,
#     "367": 19,
#     "377": 16,
#     "381": 16,
#     "382": 0,
#     "388": 17,
#     "389": 14,
#     "390": 9,
#     "395": 7,
#     "403": 0,
#     "404": 0,
#     "406": 2,
#     "413": 10,
#     "417": 7,
#     "418": 10,
#     "420": 3,
#     "422": 12,
#     "436": 0,
#     "439": 1,
#     "440": 19,
#     "451": 4,
#     "458": 5,
#     "472": 3,
#     "476": 3,
#     "477": 2,
#     "483": 15,
#     "484": 9,
#     "489": 3,
#     "490": 2,
#     "492": 0,
#     "312": 2,
#     "315": 2,
#     "317": 8,
#     "322": 5,
#     "324": 5,
#     "352": 10,
#     "353": 11,
#     "355": 10,
#     "356": 10,
#     "357": 7,
#     "358": 7,
#     "360": 4,
#     "363": 0,
#     "364": 0,
#     "366": 0,
#     "368": 7,
#     "369": 0,
#     "370": 0,
#     "376": 12,
#     "380": 10,
#     "386": 11,
#     "402": 11,
#     "409": 10,
#     "421": 10,
#     "426": 20,
#     "433": 10,
#     "441": 18,
#     "448": 18,
#     "459": 16,
#     "354": 18,
#     "359": 13,
#     "365": 12,
#     "384": 15,
#     "405": 17,
#     "410": 12,
#     "453": 17,
#     "461": 17,
# }
# # Create directory for saving models
# os.makedirs(MODEL_SAVE_PATH, exist_ok=True)


# # ========================
# # 2. DATA PREPROCESSING
# # ========================
# def process_audio(audio_path, target_sr, duration):
#     try:
#         # Load audio with target sample rate
#         audio, sr = librosa.load(audio_path, sr=target_sr)

#         # Trim or pad audio to fixed duration
#         if len(audio) < target_sr * duration:
#             audio = np.pad(
#                 audio, (0, max(0, target_sr * duration - len(audio))), "constant"
#             )
#         else:
#             audio = audio[: int(target_sr * duration)]  # Ensure integer for slicing

#         # Remove silence (less aggressive than before)
#         audio, _ = librosa.effects.trim(audio, top_db=25)

#         # Check for sufficient audio after trimming
#         if len(audio) < sr * 0.5:  # Require at least 0.5 seconds of audio
#             print(f"Audio too short after trimming: {audio_path}")
#             return None

#         # Enhanced normalization and pre-emphasis
#         audio = librosa.effects.preemphasis(audio)
#         audio = librosa.util.normalize(audio)

#         return audio  # Assuming you want to return the processed audio

#     except Exception as e:
#         print(f"An error occurred during audio processing for {audio_path}: {e}")
#         return None  # Or handle the error as appropriate for your application


# # ========================
# # 3. FEATURE EXTRACTION (ENHANCED)
# # ========================
# def voice_activity_features(audio, sr):
#     """Calculate voice activity detection features"""
#     rms = librosa.feature.rms(y=audio)
#     vad = rms > np.percentile(rms, 30)
#     return {
#         "vad_ratio": np.mean(vad),
#         "vad_transitions": np.sum(np.diff(vad.astype(int)) != 0),
#     }


# def jitter_shimmer(audio, sr):
#     """Calculate voice quality metrics"""
#     pitches = librosa.yin(audio, fmin=80, fmax=400)
#     valid_pitches = pitches[pitches > 0]
#     if len(valid_pitches) > 1:
#         jitter = np.mean(np.abs(np.diff(valid_pitches))) / np.mean(valid_pitches)
#         return {"jitter": jitter}
#     return {"jitter": 0}


# def extract_features(audio_path):
#     """Enhanced feature extraction with new voice quality metrics"""
#     base_name = os.path.basename(audio_path).split("_")[0]
#     features = {"participant_id": base_name}

#     # Load and preprocess audio
#     audio = process_audio(audio_path, target_sr=16000, duration=5)
#     sr = 16000
#     if audio is None:
#         return None

#     # 1. Acoustic features
#     # MFCCs (with more detailed statistics)
#     mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
#     for i in range(20):
#         features.update(
#             {
#                 f"mfcc_{i}_mean": np.mean(mfccs[i]),
#                 f"mfcc_{i}_std": np.std(mfccs[i]),
#                 f"mfcc_{i}_kurtosis": pd.Series(mfccs[i]).kurtosis(),
#             }
#         )

#     # Chroma features
#     chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
#     features.update({f"chroma_{i}_mean": np.mean(chroma[i]) for i in range(12)})

#     # Spectral features
#     spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
#     spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
#     spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
#     spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
#     features.update(
#         {
#             "spectral_centroid_mean": np.mean(spectral_centroid),
#             "spectral_bandwidth_mean": np.mean(spectral_bandwidth),
#             "spectral_rolloff_mean": np.mean(spectral_rolloff),
#             "spectral_contrast_mean": np.mean(spectral_contrast),
#         }
#     )

#     # 2. Prosodic features (enhanced)
#     pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
#     valid_pitches = pitches[pitches > 0]
#     pitch_mean = np.mean(valid_pitches) if len(valid_pitches) > 0 else 0

#     features.update(
#         {
#             "pitch_mean": pitch_mean,
#             "pitch_std": np.std(valid_pitches) if len(valid_pitches) > 0 else 0,
#             "harmonic_ratio": np.mean(librosa.effects.harmonic(audio)),
#             "percussive_ratio": np.mean(librosa.effects.percussive(audio)),
#         }
#     )

#     # Add new voice quality features
#     features.update(voice_activity_features(audio, sr))
#     features.update(jitter_shimmer(audio, sr))

#     # Speaking rate and pauses (more robust calculation)
#     intervals = librosa.effects.split(audio, top_db=25)
#     speech_duration = sum(end - start for start, end in intervals) / sr
#     total_duration = len(audio) / sr
#     features.update(
#         {
#             "pause_ratio": (
#                 (total_duration - speech_duration) / total_duration
#                 if total_duration > 0
#                 else 0
#             ),
#             "speaking_rate": (
#                 len(intervals) / total_duration if total_duration > 0 else 0
#             ),
#             "speech_rate_var": (
#                 np.std([(end - start) / sr for start, end in intervals])
#                 if len(intervals) > 1
#                 else 0
#             ),
#         }
#     )

#     # 3. Linguistic features (from transcript)
#     transcript_path = os.path.join(DATASET_PATH, f"{base_name}_TRANSCRIPT.csv")
#     if os.path.exists(transcript_path):
#         try:
#             transcript = pd.read_csv(transcript_path, sep="\t")
#             text = " ".join(str(x) for x in transcript["value"].dropna())
#             blob = TextBlob(text)
#             features.update(
#                 {
#                     "sentiment_polarity": blob.sentiment.polarity,
#                     "sentiment_subjectivity": blob.sentiment.subjectivity,
#                     "word_count": len(text.split()),
#                     "avg_word_length": (
#                         np.mean([len(word) for word in text.split()]) if text else 0
#                     ),
#                     "lexical_diversity": (
#                         len(set(text.split())) / len(text.split()) if text else 0
#                     ),
#                     "negative_word_ratio": (
#                         sum(
#                             1
#                             for word in text.split()
#                             if blob.words[0].sentiment.polarity < -0.1
#                         )
#                         / len(text.split())
#                         if text
#                         else 0
#                     ),
#                 }
#             )
#         except Exception as e:
#             print(f"Transcript error for {base_name}: {str(e)}")

#     return features


# # ========================
# # 4. PROCESS ALL FILES
# # ========================
# def process_all_files():
#     all_features = []
#     audio_files = glob.glob(os.path.join(DATASET_PATH, "*_AUDIO.wav"))

#     for audio_path in audio_files:
#         print(f"Processing {os.path.basename(audio_path)}...")
#         features = extract_features(audio_path)
#         if features is not None:
#             all_features.append(features)

#     # Convert to DataFrame
#     features_df = pd.DataFrame(all_features)

#     # Add labels if available
#     if PHQ8_SCORES:
#         features_df["phq8_score"] = features_df["participant_id"].map(PHQ8_SCORES)
#         features_df["depression_label"] = (features_df["phq8_score"] >= 10).astype(int)

#     # Save to CSV
#     features_df.to_csv(OUTPUT_CSV, index=False)
#     print(f"\nSaved features for {len(all_features)} participants to {OUTPUT_CSV}")
#     return features_df


# # ========================
# # 5. FEATURE SELECTION & PREPARATION
# # ========================
# def prepare_features(features_df):
#     # Separate features and labels
#     X = features_df.drop(
#         columns=["participant_id", "phq8_score", "depression_label"], errors="ignore"
#     )
#     y = features_df.get("depression_label", None)

#     # Handle missing values
#     imputer = SimpleImputer(strategy="mean")
#     X_imputed = imputer.fit_transform(X)

#     # Feature scaling
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X_imputed)

#     # Feature selection (only if we have enough samples and labels)
#     if y is not None and X_scaled.shape[0] > 1:
#         selector = SelectKBest(f_classif, k=min(30, X_scaled.shape[1]))
#         X_selected = selector.fit_transform(X_scaled, y)
#         selected_features = X.columns[selector.get_support()]
#         print(f"Selected top features: {selected_features.tolist()}")
#     else:
#         X_selected = X_scaled
#         selected_features = (
#             X.columns
#         )  # If no selection, all original features are "selected"

#     return (
#         X_selected,
#         y,
#         selected_features,
#     )  # Ensure selected_features is always returned


# # ========================
# # 6. MODEL TRAINING (ENHANCED)
# # ========================
# def train_models(X, y, feature_names):
#     from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
#     from sklearn.calibration import CalibratedClassifierCV
#     from sklearn.pipeline import make_pipeline
#     import pandas as pd

#     if y is None:
#         print("No labels available for training.")
#         return None

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )

#     class_weights = compute_class_weight(
#         "balanced", classes=np.unique(y_train), y=y_train
#     )
#     class_weight_dict = dict(zip(np.unique(y_train), class_weights))

#     results = {}

#     models = {
#         "random_forest": RandomForestClassifier(
#             n_estimators=300,
#             class_weight=class_weight_dict,
#             max_depth=10,
#             min_samples_split=5,
#             random_state=42,
#         ),
#         "svm": make_pipeline(
#             StandardScaler(),
#             CalibratedClassifierCV(
#                 SVC(
#                     kernel="rbf",
#                     class_weight="balanced",
#                     probability=True,
#                     gamma="auto",
#                     C=0.1,
#                     random_state=42,
#                 ),
#                 method="sigmoid",
#                 cv=3,
#             ),
#         ),
#         "xgboost": xgb.XGBClassifier(
#             objective="binary:logistic",
#             eval_metric="logloss",
#             use_label_encoder=False,
#             scale_pos_weight=sum(y_train == 0) / sum(y_train == 1),
#             n_estimators=200,
#             max_depth=6,
#             learning_rate=0.05,
#             subsample=0.8,
#             colsample_bytree=0.8,
#             random_state=42,
#         ),
#     }

#     # Train and save each model with features
#     for name, model in models.items():
#         print(f"\nTraining {name} ...")
#         try:
#             # Cross-validation
#             cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
#             print(f"CV ROC-AUC: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}")
            
#             # Fit model
#             model.fit(X_train, y_train)
            
#             # Save the model
#             model_filename = f"{name}_model.joblib"
#             model_path = os.path.join(MODEL_SAVE_PATH, model_filename)
#             dump(model, model_path)
#             print(f"✅ Saved {name} model to {model_path}")

#             # Get feature importances
#             feature_imp = get_feature_importances(model, feature_names)
            
#             # Save feature importances with specific naming
#             if feature_imp is not None:
#                 if name == "svm":
#                     features_filename = "svm_features.joblib"
#                 else:
#                     features_filename = f"{name}_features.joblib"
                
#                 features_path = os.path.join(MODEL_SAVE_PATH, features_filename)
#                 dump(feature_imp, features_path)
#                 print(f"✅ Saved {name} features to {features_path}")

#             # Evaluate model
#             y_pred = model.predict(X_test)
#             y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

#             results[name] = {
#                 "model": model,
#                 "accuracy": accuracy_score(y_test, y_pred),
#                 "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
#                 "avg_precision": average_precision_score(y_test, y_proba) if y_proba is not None else None,
#                 "report": classification_report(y_test, y_pred, output_dict=True),
#                 "confusion_matrix": confusion_matrix(y_test, y_pred),
#                 "feature_importances": feature_imp,
#             }

#             # Plot and save confusion matrix
#             plot_confusion_matrix(results[name]["confusion_matrix"], f"{name} Confusion Matrix")

#         except Exception as e:
#             print(f"⚠️ Error training {name}: {str(e)}")
#             continue

#     # Train and save LightGBM model with features
#     print("\nTraining LightGBM...")
#     scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)

#     lgb_model = lgb.LGBMClassifier(
#         objective="binary",
#         metric="auc",
#         boosting_type="gbdt",
#         feature_pre_filter=False,
#         min_data_in_leaf=max(1, int(0.05 * len(y_train))),
#         min_data_in_bin=1,
#         scale_pos_weight=scale_pos_weight,
#         n_jobs=-1,
#         random_state=42,
#     )

#     # Hyperparameter tuning
#     if len(y_train) >= 60:
#         param_grid = {
#             "n_estimators": [100, 200],
#             "learning_rate": [0.01, 0.05, 0.1],
#             "max_depth": [-1, 4, 6],
#             "num_leaves": [7, 15, 31],
#             "subsample": [0.8, 1.0],
#             "colsample_bytree": [0.8, 1.0],
#             "reg_alpha": [0.0, 0.1],
#             "reg_lambda": [0.0, 1.0],
#         }
#         grid = GridSearchCV(lgb_model, param_grid, cv=3, scoring="roc_auc", n_jobs=-1)
#         grid.fit(X_train, y_train)
#         best_lgb_model = grid.best_estimator_
#     else:
#         best_lgb_model = lgb_model.set_params(
#             n_estimators=200,
#             learning_rate=0.05,
#             max_depth=-1,
#             num_leaves=31,
#             subsample=0.8,
#             colsample_bytree=0.8,
#             reg_alpha=0.1,
#             reg_lambda=1.0,
#         ).fit(X_train, y_train)

#     # Calibration
#     calibrated_lgb = CalibratedClassifierCV(best_lgb_model, method="isotonic", cv=3) if len(y_train) >= 90 else best_lgb_model
#     calibrated_lgb.fit(X_train, y_train)
    
#     # Save LightGBM model
#     lgb_model_path = os.path.join(MODEL_SAVE_PATH, "lightgbm_model.joblib")
#     dump(calibrated_lgb, lgb_model_path)
#     print(f"✅ Saved LightGBM model to {lgb_model_path}")

#     # Get and save LightGBM features
#     lgb_feature_imp = get_feature_importances(best_lgb_model, feature_names)
#     if lgb_feature_imp is not None:
#         lgb_features_path = os.path.join(MODEL_SAVE_PATH, "lightgbm_features.joblib")
#         dump(lgb_feature_imp, lgb_features_path)
#         print(f"✅ Saved LightGBM features to {lgb_features_path}")

#     # Evaluate LightGBM
#     y_pred = calibrated_lgb.predict(X_test)
#     y_proba = calibrated_lgb.predict_proba(X_test)[:, 1]

#     results["lightgbm"] = {
#         "model": calibrated_lgb,
#         "accuracy": accuracy_score(y_test, y_pred),
#         "roc_auc": roc_auc_score(y_test, y_proba),
#         "avg_precision": average_precision_score(y_test, y_proba),
#         "report": classification_report(y_test, y_pred, output_dict=True),
#         "confusion_matrix": confusion_matrix(y_test, y_pred),
#         "feature_importances": lgb_feature_imp,
#     }

#     # Plot and save LightGBM confusion matrix
#     plot_confusion_matrix(results["lightgbm"]["confusion_matrix"], "LightGBM Confusion Matrix")

#     # Create and save ensemble model if we have multiple models
#     successful_models = [(name, results[name]["model"]) for name in results if "model" in results[name]]
#     if len(successful_models) > 1:
#         print("\nCreating ensemble model...")
#         ensemble = VotingClassifier(
#             estimators=successful_models,
#             voting="soft"
#         )
#         ensemble.fit(X_train, y_train)
        
#         # Save ensemble model
#         ensemble_path = os.path.join(MODEL_SAVE_PATH, "ensemble_model.joblib")
#         dump(ensemble, ensemble_path)
#         print(f"✅ Saved ensemble model to {ensemble_path}")
        
#         # Evaluate ensemble
#         y_pred = ensemble.predict(X_test)
#         y_proba = ensemble.predict_proba(X_test)[:, 1]
        
#         results["ensemble"] = {
#             "model": ensemble,
#             "accuracy": accuracy_score(y_test, y_pred),
#             "roc_auc": roc_auc_score(y_test, y_proba),
#             "avg_precision": average_precision_score(y_test, y_proba),
#             "report": classification_report(y_test, y_pred, output_dict=True),
#             "confusion_matrix": confusion_matrix(y_test, y_pred)
#         }
        
#         # Plot and save ensemble confusion matrix
#         plot_confusion_matrix(results["ensemble"]["confusion_matrix"], "Ensemble Confusion Matrix")

#     # Save feature names (both overall_selected_features and selected_features)
#     overall_features_path = os.path.join(MODEL_SAVE_PATH, "overall_selected_features.joblib")
#     dump(feature_names.tolist(), overall_features_path)
#     print(f"✅ Saved overall selected features to {overall_features_path}")
    
#     selected_features_path = os.path.join(MODEL_SAVE_PATH, "selected_features.joblib")
#     dump(feature_names.tolist(), selected_features_path)
#     print(f"✅ Saved selected features to {selected_features_path}")

#     # Save training results
#     results_path = os.path.join(MODEL_SAVE_PATH, "training_results.joblib")
#     dump(results, results_path)
#     print(f"✅ Saved training results to {results_path}")

#     return results

# def plot_confusion_matrix(cm, title='Confusion Matrix'):
#     """Plot a confusion matrix with proper formatting"""
#     plt.figure(figsize=(6, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                xticklabels=['Not Depressed', 'Depressed'],
#                yticklabels=['Not Depressed', 'Depressed'])
#     plt.title(title)
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.tight_layout()
#     plt.savefig(f"{title.lower().replace(' ', '_')}_confusion_matrix.png")
#     plt.show()


# def get_feature_importances(model, feature_names):
#     """Extract feature importances in a readable format"""
#     if hasattr(model, "feature_importances_"):
#         return sorted(
#             zip(feature_names, model.feature_importances_),
#             key=lambda x: x[1],
#             reverse=True,
#         )
#     elif (
#         hasattr(model, "named_steps") and "calibratedclassifiercv" in model.named_steps
#     ):
#         # For SVM which is a pipeline with SelectKBest and then SVC
#         # We need to get the feature importances from the SVC's coefficients
#         # and map them back to the features selected by the inner SelectKBest.
#         # This part assumes the inner SelectKBest is named 'selectkbest'
#         if "selectkbest" in model.named_steps and hasattr(
#             model.named_steps["calibratedclassifiercv"].estimator, "coef_"
#         ):
#             coef = model.named_steps["calibratedclassifiercv"].estimator.coef_[0]
#             selector = model.named_steps["selectkbest"]
#             selected_mask = selector.get_support()
#             # The feature_names passed here are the 30 features from the outer selection
#             # We need to map coef to these 30 features.
#             # This is complex because the inner SelectKBest selects from these 30.
#             # For simplicity, we'll return None for SVM feature importances if it's too complex to map directly.
#             return None  # Or implement more complex mapping if critical
#     return None


# # ========================
# # 7. VISUALIZATION (ENHANCED)
# # ========================
# def visualize_features(features_df):
#     if "depression_label" not in features_df.columns:
#         print("No labels available for visualization")
#         return

#     plt.figure(figsize=(20, 15))

#     # 1. Voice Quality Features
#     plt.subplot(2, 2, 1)
#     if "jitter" in features_df.columns:
#         sns.boxplot(data=features_df, x="depression_label", y="jitter")
#         plt.title("Jitter by Depression Status")

#     # 2. Prosodic Features
#     plt.subplot(2, 2, 2)
#     if "pause_ratio" in features_df.columns and "speaking_rate" in features_df.columns:
#         sns.scatterplot(
#             data=features_df,
#             x="pause_ratio",
#             y="speaking_rate",
#             hue="depression_label",
#             style="depression_label",
#         )
#         plt.title("Pause Ratio vs Speaking Rate")

#     # 3. MFCC Features
#     plt.subplot(2, 2, 3)
#     mfcc_means = [
#         col for col in features_df.columns if col.startswith("mfcc_") and "_mean" in col
#     ][:5]
#     if mfcc_means:
#         mfcc_data = pd.melt(
#             features_df[["depression_label"] + mfcc_means], id_vars="depression_label"
#         )
#         sns.boxplot(data=mfcc_data, x="variable", y="value", hue="depression_label")
#         plt.title("MFCC Means Distribution")
#         plt.xticks(rotation=45)

#     # 4. Feature Correlation (only include available features)
#     plt.subplot(2, 2, 4)
#     possible_features = [
#         "spectral_centroid_mean",
#         "pitch_mean",
#         "jitter",
#         "pause_ratio",
#         "sentiment_polarity",
#         "depression_label",
#     ]
#     available_features = [f for f in possible_features if f in features_df.columns]

#     if len(available_features) > 1:  # Need at least 2 features for correlation
#         sns.heatmap(features_df[available_features].corr(), annot=True, cmap="coolwarm")
#         plt.title("Feature Correlation Matrix")
#     else:
#         plt.text(
#             0.5,
#             0.5,
#             "Not enough features for correlation matrix",
#             ha="center",
#             va="center",
#         )
#         plt.title("Feature Correlation (Not Available)")

#     plt.tight_layout()
#     plt.savefig("feature_analysis.png")
#     plt.show()


# # ========================
# # 8. MAIN EXECUTION
# # ========================
# if __name__ == "__main__":
#     # Step 1: Process all files and extract features
#     print("Processing files and extracting features...")
#     features_df = process_all_files()

#     # Step 2: Visualize features
#     print("\nVisualizing features...")
#     visualize_features(features_df)

#     # Step 3: Prepare features for ML
#     print("\nPreparing features for machine learning...")
#     X, y, feature_names = prepare_features(
#         features_df
#     )  # feature_names will be the selected 30

#     # Step 4: Train models (if labels available)
#     if y is not None:
#         print("\nTraining machine learning models...")
#         results = train_models(X, y, feature_names)
#         selected_features = joblib.load(
#             "trained_models/overall_selected_features.joblib"
#         )
#         print("\nSelected 30 Features:")
#         for i, feat in enumerate(selected_features, 1):
#             print(f"{i}. {feat}")

#         # Print feature importances
#         for name, res in results.items():
#             if res.get("feature_importances"):
#                 print(f"\n{name} Top Features:")
#                 for feat, imp in res["feature_importances"][:10]:
#                     print(f"{feat}: {imp:.4f}")
#     else:
#         print("\nNo labels available. Skipping model training.")
#         print(
#             "Please add PHQ-8 scores to the PHQ8_SCORES dictionary to enable model training."
#         )



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

def extract_features(file_path, file_id):
    """Extracts a comprehensive set of audio features from a single file."""
    features = {}
    
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
        
        # Add metadata for labeling
        features['participant_id'] = file_id
        features['phq8_score'] = PHQ8_SCORES.get(file_id)
        
        return features
        
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

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
    lgbm = make_pipeline(StandardScaler(), lgb.LGBMClassifier(random_state=42, class_weight='balanced'))
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

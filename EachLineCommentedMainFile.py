# ========================
# 1. CONFIGURATION
# ========================
# These lines configure the script by defining constants.
import os # Imports the 'os' module for interacting with the operating system, like creating directories.
import glob # Imports the 'glob' module to find all files matching a specified pattern.
import pandas as pd # Imports the 'pandas' library and aliases it as 'pd' for data manipulation and analysis.
import librosa # Imports the 'librosa' library for audio analysis.
import numpy as np # Imports the 'numpy' library and aliases it as 'np' for numerical operations, especially with arrays.
from textblob import TextBlob # Imports the 'TextBlob' class for natural language processing, specifically for sentiment analysis.
import matplotlib.pyplot as plt # Imports 'matplotlib.pyplot' for creating plots and visualizations.
import seaborn as sns # Imports 'seaborn' for creating aesthetically pleasing statistical visualizations.
from sklearn.model_selection import train_test_split, RandomizedSearchCV # Imports functions from 'sklearn' for splitting data and hyperparameter tuning.
from sklearn.preprocessing import StandardScaler # Imports 'StandardScaler' to standardize features by removing the mean and scaling to unit variance.
from sklearn.ensemble import RandomForestClassifier # Imports the 'RandomForestClassifier' for a tree-based ensemble model.
from sklearn.svm import SVC # Imports 'SVC' for Support Vector Machine classification.
from sklearn.metrics import ( # Imports various metrics from 'sklearn' to evaluate model performance.
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    ConfusionMatrixDisplay
)
from sklearn.pipeline import make_pipeline # Imports 'make_pipeline' to create a sequence of data processing and modeling steps.
from sklearn.impute import SimpleImputer # Imports 'SimpleImputer' to handle missing data.
from sklearn.decomposition import PCA # Imports 'PCA' for dimensionality reduction.
from sklearn.feature_selection import SelectKBest, f_classif # Imports 'SelectKBest' for feature selection and 'f_classif' as the scoring function.
from sklearn.calibration import CalibratedClassifierCV # Imports 'CalibratedClassifierCV' to calibrate classifier probabilities.
from sklearn.utils.class_weight import compute_class_weight # Imports 'compute_class_weight' to handle class imbalance.
from joblib import dump, load # Imports 'dump' and 'load' from 'joblib' to save and load Python objects, including trained models.
import lightgbm as lgb # Imports the 'lightgbm' library and aliases it as 'lgb' for a gradient boosting framework.
import xgboost as xgb # Imports the 'xgboost' library and aliases it as 'xgb' for another gradient boosting framework.
from sklearn.ensemble import StackingClassifier # Imports 'StackingClassifier' for creating an ensemble model.
import joblib # Imports the 'joblib' library to save and load models efficiently.
from catboost import CatBoostClassifier # Imports 'CatBoostClassifier' for a gradient boosting library.

# Suppress warnings
import warnings # Imports the 'warnings' module.
warnings.filterwarnings("ignore") # This line suppresses all warning messages during script execution to keep the output clean.

# ========================
# 1. CONFIGURATION
# ========================
DATASET_PATH = "Dataset" # Sets the path to the directory containing the audio files.
OUTPUT_CSV = "all_features.csv" # Defines the filename for the CSV where extracted features will be saved.
MODEL_SAVE_PATH = "trained_models" # Defines the directory to save the trained models.
TRANSCRIPTS_AVAILABLE = None # Initializes a variable to check for the presence of transcript files.

# PHQ-8 scores for depression classification (0-24 range)
# A score of >= 10 is generally considered a strong indicator of depression.
# '1' will now represent a score >= 10, and '0' will represent a score < 10.
PHQ8_SCORES = { # A dictionary mapping participant IDs to their PHQ-8 depression scores.
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
os.makedirs(MODEL_SAVE_PATH, exist_ok=True) # This line creates the 'trained_models' directory if it doesn't already exist.

# ========================
# 2. DATA PREPROCESSING
# ========================
def process_audio(audio_path, target_sr, duration): # Defines a function to preprocess a single audio file.
    """
    Processes the audio file, including trimming/padding, silence removal,
    and normalization.
    """
    try: # Starts a 'try-except' block to handle potential errors during audio processing.
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True) # Loads the audio file using 'librosa', resampling it to 'target_sr' and ensuring it's a single channel ('mono').
        
        if len(audio) < target_sr * duration: # Checks if the audio is shorter than the desired duration.
            audio = np.pad(audio, (0, max(0, int(target_sr * duration) - len(audio))), 'constant') # If so, it pads the audio with zeros to reach the specified duration.
        else: # Otherwise,
            audio = audio[:int(target_sr * duration)] # It trims the audio to the specified duration.
            
        audio, _ = librosa.effects.trim(audio, top_db=25) # Removes leading and trailing silence from the audio.
        
        if len(audio) < sr * 0.5: # Checks if the audio is too short after trimming.
            raise ValueError("Audio too short after trimming.") # Raises an error if the audio is too short.
            
        audio = librosa.effects.preemphasis(audio) # Applies pre-emphasis to the audio to boost high frequencies.
        audio = librosa.util.normalize(audio) # Normalizes the audio to a standard range.
        
        return audio, sr # Returns the processed audio and its sample rate.
    except Exception as e: # Catches any exception that occurred.
        print(f"Audio processing error for {audio_path}: {e}") # Prints an error message with the file path and the error.
        return None, None # Returns 'None' for both audio and sample rate, indicating failure.

def voice_activity_features(audio, sr): # Defines a function to calculate voice activity detection (VAD) features.
    """Calculates voice activity detection features."""
    rms = librosa.feature.rms(y=audio) # Computes the root-mean-square (RMS) energy of the audio, a proxy for loudness.
    vad = rms > np.percentile(rms, 30) # Creates a boolean array where 'True' indicates speech (RMS above the 30th percentile) and 'False' indicates silence.
    return { # Returns a dictionary of VAD features.
        'vad_ratio': np.mean(vad), # The proportion of time the audio is considered to be speech.
        'vad_transitions': np.sum(np.diff(vad.astype(int)) != 0) # The number of times the audio transitions between speech and silence.
    }

def calculate_jitter(audio, sr): # Defines a function to calculate jitter, a measure of pitch variation.
    """Calculates jitter based on pitch changes."""
    pitches = librosa.yin(audio, fmin=80, fmax=400) # Extracts the fundamental frequency (pitch) of the audio using the YIN algorithm.
    valid_pitches = pitches[pitches > 0] # Filters out zero-valued pitches (where no pitch was detected).
    if len(valid_pitches) > 1: # Checks if there are enough valid pitches to calculate jitter.
        return np.mean(np.abs(np.diff(valid_pitches))) / np.mean(valid_pitches) # Calculates the mean absolute difference between consecutive pitches, normalized by the mean pitch.
    return 0 # Returns 0 if there aren't enough valid pitches.
def detect_transcripts(): # Defines a function to check for the presence of transcript files.
    """Check if any transcript files exist in the dataset."""
    transcript_files = glob.glob(os.path.join(DATASET_PATH, "*_TRANSCRIPT.csv")) # Finds all files ending in '_TRANSCRIPT.csv' within the dataset path.
    return len(transcript_files) > 0 # Returns 'True' if at least one such file is found, 'False' otherwise.
def extract_features(file_path, file_id): # Defines the main feature extraction function.
    """Extracts a comprehensive set of audio features from a single file."""
    features = {} # Initializes an empty dictionary to store the extracted features.
    global TRANSCRIPTS_AVAILABLE # Declares that the function will use the global variable 'TRANSCRIPTS_AVAILABLE'.
    
    try: # Starts a 'try-except' block.
        audio, sr = process_audio(file_path, target_sr=16000, duration=5) # Calls the 'process_audio' function to preprocess the audio file.
        if audio is None: # Checks if the audio processing failed.
            return None # Returns 'None' if the processing failed.

        # --- Acoustic Features ---
        n_mfcc_to_use = 20 # Sets the number of MFCCs to use.
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40) # Extracts 40 Mel-frequency cepstral coefficients (MFCCs).
        for i in range(n_mfcc_to_use): # Iterates through the first 20 MFCCs.
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i]) # Calculates and stores the mean of each MFCC.
            features[f'mfcc_{i}_std'] = np.std(mfccs[i]) # Calculates and stores the standard deviation of each MFCC.
            features[f'mfcc_{i}_kurtosis'] = pd.Series(mfccs[i]).kurtosis() # Calculates and stores the kurtosis of each MFCC.

        chroma = librosa.feature.chroma_stft(y=audio, sr=sr) # Extracts chroma features, which represent the energy in each of the 12 pitch classes.
        for i in range(12): # Iterates through the 12 chroma features.
            features[f'chroma_{i}_mean'] = np.mean(chroma[i]) # Stores the mean of each chroma feature.
        
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr) # Extracts the spectral centroid, which indicates the "center of mass" of the audio spectrum.
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr) # Extracts the spectral bandwidth, which measures the width of the spectrum.
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr) # Extracts the spectral rolloff, the frequency below which a certain percentage of the total spectral energy lies.
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr) # Extracts the spectral contrast, which measures the difference in energy between peaks and valleys in the spectrum.
        features.update({ # Adds the mean values of these spectral features to the 'features' dictionary.
            "spectral_centroid_mean": np.mean(spectral_centroid),
            "spectral_bandwidth_mean": np.mean(spectral_bandwidth),
            "spectral_rolloff_mean": np.mean(spectral_rolloff),
            "spectral_contrast_mean": np.mean(spectral_contrast),
        })

        # --- Prosodic Features ---
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr) # Extracts pitches and their magnitudes from the audio.
        valid_pitches = pitches[pitches > 0] # Filters for valid pitch values.
        pitch_mean = np.mean(valid_pitches) if len(valid_pitches) > 0 else 0 # Calculates the mean pitch, handling the case where no pitches are found.
        
        features.update({ # Adds a set of prosodic features to the dictionary.
            "pitch_mean": pitch_mean,
            "pitch_std": np.std(valid_pitches) if len(valid_pitches) > 0 else 0, # Standard deviation of pitch.
            "harmonic_ratio": np.mean(librosa.effects.harmonic(audio)), # The ratio of harmonic to noise components in the audio.
            "percussive_ratio": np.mean(librosa.effects.percussive(audio)), # The ratio of percussive to harmonic components.
        })

        features.update(voice_activity_features(audio, sr)) # Adds the VAD features calculated earlier.
        features['jitter'] = calculate_jitter(audio, sr) # Adds the jitter feature.

        intervals = librosa.effects.split(audio, top_db=25) # Splits the audio into non-silent intervals.
        speech_duration = sum(end-start for start,end in intervals)/sr # Calculates the total duration of speech.
        total_duration = len(audio)/sr # Calculates the total duration of the audio.
        features.update({ # Adds speech-related timing features.
            "pause_ratio": (total_duration - speech_duration) / total_duration if total_duration > 0 else 0, # The proportion of time spent in silence.
            "speaking_rate": len(intervals) / total_duration if total_duration > 0 else 0, # The number of speech intervals per second.
            "speech_rate_var": np.std([(end-start)/sr for start,end in intervals]) if len(intervals) > 1 else 0, # The variation in the duration of speech intervals.
        })

        # --- Linguistic features (requires transcripts, which are not provided) ---
        # The following block is commented out and will not be executed in this version of the code due to the `TRANSCRIPTS_AVAILABLE` flag being `None`.
        # It's a placeholder for future functionality.
        if TRANSCRIPTS_AVAILABLE: # This condition will always be false with the current configuration.
            transcript_path = os.path.join(DATASET_PATH, f"{file_id}_TRANSCRIPT.csv")
            print('Transcripts Available',transcript_path)

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
        features['participant_id'] = file_id # Adds the participant ID to the features.
        features['phq8_score'] = PHQ8_SCORES.get(file_id) # Adds the PHQ-8 score from the predefined dictionary.
        
        return features # Returns the dictionary of features.
        
    except Exception as e: # Catches any exception.
        print(f"Error extracting features from {file_path}: {e}") # Prints an error message.
        return None # Returns 'None' to indicate failure.

def neutral_linguistic_features(): # Defines a function that returns a dictionary of placeholder values for linguistic features.
    """Return zero/neutral values for linguistic features."""
    return { # The dictionary of features with neutral values.
        "sentiment_polarity": 0.0,
        "sentiment_subjectivity": 0.0,
        "word_count": 0,
        "avg_word_length": 0.0,
        "lexical_diversity": 0.0,
        "negative_word_ratio": 0.0,
    }
def process_all_files(): # Defines a function to process all audio files in the dataset.
    """Goes through the dataset, extracts features, and stores them in a DataFrame."""
    feature_list = [] # Initializes an empty list to store feature dictionaries.
    file_paths = glob.glob(os.path.join(DATASET_PATH, "**", "*.wav"), recursive=True) # Finds all '.wav' files in the dataset path and its subdirectories.
    
    for file_path in file_paths: # Loops through each file path.
        file_id = os.path.basename(file_path).split('_')[0] # Extracts the participant ID from the filename.
        if file_id in PHQ8_SCORES: # Checks if the participant ID has a PHQ-8 score.
            features = extract_features(file_path, file_id) # Calls 'extract_features' for the current file.
            if features: # If features were successfully extracted,
                feature_list.append(features) # Appends the feature dictionary to the list.
    
    features_df = pd.DataFrame(feature_list) # Creates a pandas DataFrame from the list of feature dictionaries.
    features_df.to_csv(OUTPUT_CSV, index=False) # Saves the DataFrame to a CSV file without the index.
    return features_df # Returns the DataFrame.

def visualize_features(df): # Defines a function to visualize the extracted features.
    """
    Creates visualizations of the extracted features to identify trends.
    This helps in feature selection and understanding the data.
    """
    # Create the binary label for depression based on PHQ-8 score >= 10
    df['depression_label'] = df['phq8_score'].apply(lambda x: 1 if x >= 10 else 0) # Creates a new column 'depression_label' where 1 means depressed (score >= 10) and 0 means not.
    
    print("\nVisualizing top 10 features by correlation...") # Prints a message.
    
    # Calculate correlation matrix
    correlations = df.corr(numeric_only=True)['depression_label'].sort_values(ascending=False) # Calculates the correlation of all numeric features with 'depression_label' and sorts them.
    top_10_correlated_features = correlations.index[1:11] # Selects the names of the top 10 most correlated features (the first one is 'depression_label' itself, so we skip it).
    
    plt.figure(figsize=(15, 10)) # Creates a new figure for the plots.
    for i, feature in enumerate(top_10_correlated_features): # Loops through the top 10 features.
        plt.subplot(2, 5, i + 1) # Creates a subplot in a 2x5 grid.
        sns.boxplot(x='depression_label', y=feature, data=df) # Creates a box plot showing the distribution of the feature for each depression label.
        plt.title(f'Depression Label vs. {feature}', fontsize=8) # Sets the title of the subplot.
        plt.xticks([0, 1], ['Not Depressed', 'Depressed']) # Sets the x-axis labels.
    plt.tight_layout() # Adjusts subplot parameters for a tight layout.
    plt.show() # Displays the plot.

def prepare_features(df): # Defines a function to prepare features for machine learning.
    """
    Prepares the feature DataFrame for machine learning, including
    imputation, scaling, and feature selection.
    """
    # Create the binary label
    df['depression_label'] = df['phq8_score'].apply(lambda x: 1 if x >= 10 else 0) # Recalculates the depression label.
    
    # Separate features and labels
    feature_columns = [col for col in df.columns if col not in ['participant_id', 'phq8_score', 'depression_label']] # Creates a list of column names to be used as features.
    X = df[feature_columns] # Creates the feature matrix 'X'.
    y = df['depression_label'] # Creates the target vector 'y'.
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean') # Initializes an imputer that fills missing values with the mean of the column.
    X = pd.DataFrame(imputer.fit_transform(X), columns=feature_columns) # Fits the imputer and transforms the data, then converts it back to a DataFrame.
    
    # Select top K best features using ANOVA F-value
    selector = SelectKBest(f_classif, k=30) # Initializes a feature selector to select the top 30 features based on ANOVA F-value.
    X_selected = selector.fit_transform(X, y) # Fits the selector and transforms the data to include only the selected features.
    selected_mask = selector.get_support() # Gets a boolean mask of the selected features.
    selected_features = X.columns[selected_mask] # Gets the names of the selected features.
    
    # Save the list of selected feature names for the API
    joblib.dump(selected_features.tolist(), os.path.join(MODEL_SAVE_PATH, 'overall_selected_features.joblib')) # Saves the list of selected feature names.
    
    return X_selected, y, selected_features # Returns the selected feature matrix, the target vector, and the names of the selected features.

def plot_confusion_matrix(y_true, y_pred, model_name): # Defines a function to plot a confusion matrix.
    """
    Plots a confusion matrix heatmap for a given model.
    """
    cm = confusion_matrix(y_true, y_pred) # Calculates the confusion matrix.
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Depressed', 'Depressed']) # Initializes a display object for the confusion matrix.
    disp.plot(cmap=plt.cm.Blues) # Plots the confusion matrix with a blue color map.
    plt.title(f'Confusion Matrix for {model_name}') # Sets the title of the plot.
    plt.show() # Displays the plot.

def train_models(X, y, feature_names): # Defines a function to train multiple models.
    """
    Trains multiple models, including an improved ensemble, and
    saves all trained models and their feature importances.
    """
    results = {} # Initializes an empty dictionary to store the training results.
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # Splits the data into training and testing sets, ensuring the class distribution is the same in both.
    
    # Calculate class weights for imbalanced data
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train) # Computes weights for each class to handle imbalance.
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)} # Converts the weights into a dictionary.

    # --- Model Pipelines with Hyperparameter Tuning ---
    
    # 1. LightGBM Classifier with RandomizedSearchCV
    lgbm = make_pipeline(StandardScaler(), lgb.LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1)) # Creates a pipeline for LightGBM with a scaler.
    lgbm_param_dist = { # Defines the hyperparameter search space for LightGBM.
        'lgbmclassifier__n_estimators': [100, 200, 300],
        'lgbmclassifier__learning_rate': [0.01, 0.05, 0.1],
        'lgbmclassifier__num_leaves': [20, 31, 40],
    }
    lgbm_gs = RandomizedSearchCV(lgbm, lgbm_param_dist, n_iter=10, cv=5, scoring='roc_auc', random_state=42) # Initializes a randomized search for the best hyperparameters.
    lgbm_gs.fit(X_train, y_train) # Fits the randomized search to the training data.
    results['LightGBM'] = {'model': lgbm_gs.best_estimator_, 'best_params': lgbm_gs.best_params_} # Stores the best model and its parameters.
    joblib.dump(lgbm_gs.best_estimator_, os.path.join(MODEL_SAVE_PATH, 'lightgbm_model.joblib')) # Saves the trained LightGBM model.
    print(f"\n✔ LightGBM model saved to {os.path.join(MODEL_SAVE_PATH, 'lightgbm_model.joblib')}") # Prints a confirmation message.
    
    # 2. Random Forest Classifier with RandomizedSearchCV
    rf = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42, class_weight='balanced')) # Creates a pipeline for Random Forest.
    rf_param_dist = { # Defines the hyperparameter search space for Random Forest.
        'randomforestclassifier__n_estimators': [100, 200, 300],
        'randomforestclassifier__max_depth': [None, 10, 20],
        'randomforestclassifier__min_samples_leaf': [1, 2, 4],
    }
    rf_gs = RandomizedSearchCV(rf, rf_param_dist, n_iter=10, cv=5, scoring='roc_auc', random_state=42) # Initializes a randomized search for Random Forest.
    rf_gs.fit(X_train, y_train) # Fits the search to the training data.
    results['RandomForest'] = {'model': rf_gs.best_estimator_, 'best_params': rf_gs.best_params_} # Stores the best model and its parameters.
    joblib.dump(rf_gs.best_estimator_, os.path.join(MODEL_SAVE_PATH, 'random_forest_model.joblib')) # Saves the Random Forest model.
    print(f"✔ RandomForest model saved to {os.path.join(MODEL_SAVE_PATH, 'random_forest_model.joblib')}") # Prints a confirmation.
    
    # 3. XGBoost Classifier with RandomizedSearchCV
    xgb_model = make_pipeline( # Creates a pipeline for XGBoost.
        StandardScaler(), 
        xgb.XGBClassifier(
            random_state=42, 
            eval_metric='logloss',
            use_label_encoder=False,
            scale_pos_weight=class_weight_dict.get(1) / class_weight_dict.get(0) if class_weight_dict.get(0) else 1 # Sets a scale for the positive class to handle imbalance.
        )
    )
    xgb_param_dist = { # Defines the hyperparameter search space for XGBoost.
        'xgbclassifier__n_estimators': [100, 200, 300],
        'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
        'xgbclassifier__max_depth': [3, 5, 7],
        'xgbclassifier__subsample': [0.8, 1.0],
    }
    xgb_gs = RandomizedSearchCV(xgb_model, xgb_param_dist, n_iter=10, cv=5, scoring='roc_auc', random_state=42) # Initializes a randomized search for XGBoost.
    xgb_gs.fit(X_train, y_train) # Fits the search to the training data.
    results['XGBoost'] = {'model': xgb_gs.best_estimator_, 'best_params': xgb_gs.best_params_} # Stores the best model and parameters.
    joblib.dump(xgb_gs.best_estimator_, os.path.join(MODEL_SAVE_PATH, 'xgboost_model.joblib')) # Saves the XGBoost model.
    print(f"✔ XGBoost model saved to {os.path.join(MODEL_SAVE_PATH, 'xgboost_model.joblib')}") # Prints a confirmation.

    # 4. Support Vector Machine (SVC) with probability calibration
    svm = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced')) # Creates a pipeline for SVC.
    svm_calibrated = CalibratedClassifierCV(svm, method='isotonic', cv=5) # Wraps the SVC in a calibrated classifier to get better probability estimates.
    svm_calibrated.fit(X_train, y_train) # Fits the calibrated SVC.
    results['SVC'] = {'model': svm_calibrated} # Stores the calibrated SVC model.
    joblib.dump(svm_calibrated, os.path.join(MODEL_SAVE_PATH, 'svm_model.joblib')) # Saves the SVC model.
    print(f"✔ SVC model saved to {os.path.join(MODEL_SAVE_PATH, 'svm_model.joblib')}") # Prints a confirmation.
    
    catboost_model = CatBoostClassifier( # Initializes a CatBoost classifier.
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        verbose=False,
        class_weights=[class_weight_dict[0], class_weight_dict[1]] # Sets class weights.
    )
    catboost_model.fit(X_train, y_train) # Fits the CatBoost model.
    results['CatBoost'] = {'model': catboost_model} # Stores the CatBoost model.
    joblib.dump(catboost_model, os.path.join(MODEL_SAVE_PATH, 'catboost_model.joblib')) # Saves the CatBoost model.
    print(f"✔ CatBoost model saved to {os.path.join(MODEL_SAVE_PATH, 'catboost_model.joblib')}") # Prints a confirmation.


    # --- Ensemble Model (StackingClassifier) ---
    estimators = [ # Defines the list of base estimators for the stacking classifier.
        ('lgbm', lgbm_gs.best_estimator_),
        ('rf', rf_gs.best_estimator_),
        ('xgb', xgb_gs.best_estimator_),
        ('svm', svm_calibrated)
    ]
    ensemble = StackingClassifier(estimators=estimators, final_estimator=xgb.XGBClassifier(random_state=42), cv=5) # Initializes the stacking classifier with the base estimators and a final estimator.
    ensemble.fit(X_train, y_train) # Fits the ensemble model.
    results['Ensemble'] = {'model': ensemble} # Stores the ensemble model.
    joblib.dump(ensemble, os.path.join(MODEL_SAVE_PATH, 'ensemble_model.joblib')) # Saves the ensemble model.
    print(f"✔ Ensemble model saved to {os.path.join(MODEL_SAVE_PATH, 'ensemble_model.joblib')}") # Prints a confirmation.

    print("\nTraining and evaluation complete. Results:") # Prints a message.
    
    # Evaluate and visualize all models
    for name, res in results.items(): # Loops through each trained model.
        model = res['model'] # Gets the model.
        y_pred = model.predict(X_test) # Makes predictions on the test set.
        y_proba = model.predict_proba(X_test)[:, 1] # Gets the predicted probabilities for the positive class.
        
        print(f"\n--- {name} Results ---") # Prints the model name.
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}") # Prints the accuracy score.
        print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}") # Prints the ROC AUC score.
        print(f"Avg Precision: {average_precision_score(y_test, y_proba):.4f}") # Prints the average precision score.
        print(classification_report(y_test, y_pred)) # Prints the detailed classification report.
        
        # Plot and show the confusion matrix
        plot_confusion_matrix(y_test, y_pred, name) # Calls the function to plot the confusion matrix.
        
        # Save feature importances for tree-based models
        if name in ['RandomForest', 'LightGBM', 'XGBoost']: # Checks if the model is a tree-based model.
            final_model = None # Initializes a variable for the final model in the pipeline.
            if hasattr(model, 'named_steps'): # Checks if the model is a pipeline.
                for step_name, step_model in model.named_steps.items(): # Loops through the steps of the pipeline.
                    if hasattr(step_model, 'predict'): # Finds the classifier step.
                        final_model = step_model
                        break
            else: # For other models that are not pipelines.
                final_model = model
            
            if final_model and hasattr(final_model, 'feature_importances_'): # Checks if the model has feature importances.
                importances = final_model.feature_importances_ # Gets the feature importances.
                feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}) # Creates a DataFrame of features and their importances.
                feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False) # Sorts the DataFrame by importance.
                
                # Save feature importances to a joblib file
                importance_filename = f'{name.lower()}_feature_importances.joblib' # Creates a filename for the importance data.
                joblib.dump(feature_importance_df.to_records(index=False).tolist(), os.path.join(MODEL_SAVE_PATH, importance_filename)) # Saves the feature importances.
                print(f"✔ Feature importances for {name} saved to {importance_filename}") # Prints a confirmation.
                
                res['feature_importances'] = feature_importance_df.to_records(index=False).tolist() # Stores the importances in the results dictionary.
            else:
                print(f"Could not get feature importances for model: {name}") # Prints a message if importances cannot be retrieved.

    return results # Returns the dictionary of all training results.

# ========================
# 8. MAIN EXECUTION
# =======================
if __name__ == "__main__": # This block of code runs only when the script is executed directly.
    # Step 1: Process all files and extract features
    # TRANSCRIPTS_AVAILABLE = detect_transcripts()
    print(f"Transcripts available: {TRANSCRIPTS_AVAILABLE}") # Prints the status of transcript availability.
    print("Processing files and extracting features...") # Prints a message.
    features_df = process_all_files() # Calls the function to extract features from all audio files.

    # Step 2: Visualize features
    print("\nVisualizing features...") # Prints a message.
    visualize_features(features_df) # Calls the function to visualize the features.

    # Step 3: Prepare features for ML
    print("\nPreparing features for machine learning...") # Prints a message.
    X, y, feature_names = prepare_features(features_df) # Calls the function to prepare the data for modeling.
    
    # Step 4: Train models (if labels available)
    if y is not None: # Checks if the target labels exist.
        print("\nTraining machine learning models...") # Prints a message.
        results = train_models(X, y, feature_names) # Calls the function to train and evaluate the models.
        
        selected_features = joblib.load("trained_models/overall_selected_features.joblib") # Loads the list of selected features.
        print("\nSelected 30 Features:") # Prints a message.
        for i, feat in enumerate(selected_features, 1): # Loops through the selected features and prints them.
            print(f"{i}. {feat}")

        # Print feature importances
        for name, res in results.items(): # Loops through the results of each model.
            if res.get("feature_importances"): # Checks if feature importances were saved for the model.
                print(f"\n{name} Top Features:") # Prints the model name.
                for feat, imp in res["feature_importances"][:10]: # Loops through the top 10 most important features.
                    print(f"{feat}: {imp:.4f}") # Prints the feature name and its importance score.
    else:
        print("No labels found, cannot train models.") # Prints a message if no labels are available.
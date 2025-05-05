import os
import glob
import pandas as pd
import librosa
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

# ========================
# 1. CONFIGURATION
# ========================
DATASET_PATH = "Dataset"
OUTPUT_CSV = "all_features.csv"

# Add PHQ-8 scores here (format: "participant_id": score)
PHQ8_SCORES = {
    "300": 2,  
    "301": 3,  
    "302": 4,  
    "303": 0,   
    "304": 6,
    "337": 10,     
    "332": 18,  
    "311": 21,   
    "309": 15,
    "308": 22
}

# ========================
# 2. DATA PREPROCESSING
# ========================
def preprocess_audio(audio_path, target_sr=16000, duration=30):
    """Load and preprocess audio file"""
    try:
        # Load audio with target sample rate
        audio, sr = librosa.load(audio_path, sr=target_sr)

        # Trim or pad audio to fixed duration
        if len(audio) < target_sr * duration:
            audio = np.pad(audio, (0, max(0, target_sr * duration - len(audio)), 'constant'))
        else:
            audio = audio[:target_sr * duration]

        # Remove silence
        audio, _ = librosa.effects.trim(audio, top_db=30)

        # Normalize audio
        audio = librosa.util.normalize(audio)

        return audio, sr
    except Exception as e:
        print(f"Error preprocessing {audio_path}: {str(e)}")
        return None, None

# ========================
# 3. FEATURE EXTRACTION
# ========================
def extract_features(audio_path):
    """Extract features from one participant's files"""
    base_name = os.path.basename(audio_path).split("_")[0]
    features = {"participant_id": base_name}

    # Load and preprocess audio
    audio, sr = preprocess_audio(audio_path)
    if audio is None:
        return None

    # 1. Acoustic features
    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    for i in range(40):
        features.update({
            f"mfcc_{i}_mean": np.mean(mfccs[i]),
            f"mfcc_{i}_std": np.std(mfccs[i]),
            f"mfcc_{i}_max": np.max(mfccs[i]),
            f"mfcc_{i}_min": np.min(mfccs[i])
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
    features.update({
        "spectral_centroid_mean": np.mean(spectral_centroid),
        "spectral_bandwidth_mean": np.mean(spectral_bandwidth),
        "spectral_rolloff_mean": np.mean(spectral_rolloff)
    })

    # 2. Prosodic features
    # Pitch and harmonic features
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    features.update({
        "pitch_mean": np.mean(pitches[pitches > 0]),
        "pitch_std": np.std(pitches[pitches > 0]),
        "harmonic_ratio": np.mean(librosa.effects.harmonic(audio))
    })

    # Speaking rate and pauses
    intervals = librosa.effects.split(audio, top_db=30)
    features.update({
        "pause_duration": sum((end-start)/sr for start,end in intervals),
        "speech_duration": sum(end-start for start,end in intervals)/sr,
        "speaking_rate": len(intervals) / (len(audio)/sr) if len(audio) > 0 else 0
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
                "lexical_diversity": len(set(text.split())) / len(text.split()) if text else 0
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
        features_df['depression_label'] = (features_df['phq8_score'] >= 10).astype(int)  # Binary classification

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

    # Adjust PCA based on number of samples
    n_samples = X_scaled.shape[0]
    n_features = X_scaled.shape[1]

    # Feature selection (only if we have enough samples and labels)
    if y is not None and n_samples > 1:
        selector = SelectKBest(f_classif, k=min(50, n_features))
        X_selected = selector.fit_transform(X_scaled, y)
        selected_features = X.columns[selector.get_support()]
        print(f"Selected top features: {selected_features.tolist()}")
    else:
        X_selected = X_scaled

    # Dimensionality reduction (only if we have enough samples)
    if n_samples > 1:
        pca = PCA(n_components=min(20, n_samples-1, n_features))
        X_final = pca.fit_transform(X_selected)
        print(f"Reduced to {pca.n_components_} principal components")
    else:
        print("Warning: Not enough samples for PCA. Using original features.")
        X_final = X_selected

    return X_final, y

# ========================
# 6. MODEL TRAINING
# ========================
# ========================
# 6. MODEL TRAINING (UPDATED)
# ========================
def train_models(X, y):
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
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=None)

    # Models to try
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, 
            class_weight='balanced', 
            random_state=42
        ),
        "SVM": SVC(
            kernel='rbf', 
            class_weight='balanced', 
            probability=True, 
            random_state=42
        )
    }
    
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Create pipeline
        pipeline = make_pipeline(
            StandardScaler(),
            model
        )
        
        try:
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Evaluate
            y_pred = pipeline.predict(X_test)
            
            # Store results
            results[name] = {
                "model": pipeline,
                "accuracy": accuracy_score(y_test, y_pred),
                "report": classification_report(y_test, y_pred),
                "confusion_matrix": confusion_matrix(y_test, y_pred)
            }
            
            # Print results
            print(f"\n{name} Results:")
            print(f"Accuracy: {results[name]['accuracy']:.2f}")
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
            
            # Only try predict_proba if we have both classes
            if len(np.unique(y_train)) > 1:
                y_proba = pipeline.predict_proba(X_test)[:, 1]
                # You can use y_proba for ROC curves or probability analysis
            
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            continue
    
    return results

# ========================
# 7. VISUALIZATION
# ========================
def visualize_features(features_df):
    if 'depression_label' not in features_df.columns:
        print("No labels available for visualization")
        return

    # Plot distributions of key features
    plt.figure(figsize=(18, 12))

    # MFCC means comparison
    plt.subplot(2, 2, 1)
    mfcc_means = features_df.filter(regex='mfcc_.*_mean').columns[:5]
    sns.boxplot(data=pd.melt(features_df[['depression_label'] + mfcc_means.tolist()],
                                id_vars='depression_label',
                                value_vars=mfcc_means),
                x='variable', y='value', hue='depression_label')
    plt.title("MFCC Means by Depression Status")
    plt.xticks(rotation=45)

    # Prosodic features
    plt.subplot(2, 2, 2)
    sns.scatterplot(data=features_df, x="speech_duration", y="pause_duration", hue="depression_label")
    plt.title("Speech vs Pause Duration by Depression Status")

    # Linguistic features
    plt.subplot(2, 2, 3)
    sns.kdeplot(data=features_df, x="sentiment_polarity", hue="depression_label", fill=True)
    plt.title("Sentiment Distribution by Depression Status")

    # Correlation heatmap
    plt.subplot(2, 2, 4)
    selected_features = features_df[['spectral_centroid_mean', 'pitch_mean',
                                     'speaking_rate', 'sentiment_polarity',
                                     'word_count', 'depression_label']]
    sns.heatmap(selected_features.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title("Feature Correlation Matrix")

    plt.tight_layout()
    plt.savefig("feature_distributions.png")
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
    X, y = prepare_features(features_df)

    # Step 4: Train models (if labels available)
    if y is not None:
        print("\nTraining machine learning models...")
        results = train_models(X, y)
    else:
        print("\nNo labels available. Skipping model training.")
        print("Please add PHQ-8 scores to the PHQ8_SCORES dictionary to enable model training.")
# AI-Powered Audio Biomarker Analysis
> **Early Detection and Monitoring of Depression and Anxiety**

This project presents a machine learning-based framework for detecting and monitoring depression and anxiety through audio biomarkers. By combining speech signal processing and AI, it provides a non-invasive, scalable, and real-time screening tool for mental health.

---

## Project Overview
Traditional depression screening relies heavily on self-reports and clinician-led interviews, which can introduce subjectivity and delay diagnosis. 

This project explores an **AI-driven audio analysis pipeline** to objectively identify vocal biomarkers associated with mental health conditions. By analyzing nuances in speech—such as pitch, rhythm, and spectral density—the system provides a data-driven layer to clinical assessments.


## Dataset
- The project uses the DAIC-WOZ dataset — part of the Distress Analysis Interview Corpus (DAIC).
- Due to data licensing, raw dataset files are not included in this repository.
- You can request access at: DAIC-WOZ Dataset Portal

## Key Features
- **Audio Preprocessing:** Automated silence trimming, normalization, and pre-emphasis filtering.
- **Feature Extraction:** Extraction of MFCCs, spectral, chroma, jitter, pause ratio, and voice activity metrics.
- **Model Training:** Implementation of LightGBM, Random Forest, XGBoost, SVM, and a Stacking Ensemble.
- **Evaluation:** Robust metrics including ROC-AUC, Precision, Recall, F1-score, and Confusion Matrices.
- **Flask API Deployment:** A web-based interface to upload short audio clips for instant risk predictions.
- **Explainable ML:** Feature importance visualization using SHAP for clinical interpretability.

## Project Structure
```text
AI-POWERED-AUDIO-BIOMARKER-ANALYSIS
│
├── feature_extraction.ipynb          # MFCCs, spectral, and prosodic feature extraction
├── main.py                           # training of models with extracted features
├── app.py                            # Flask API for deployment
├── requirements.txt                  # Project dependencies
├── static/ & templates/              # Web interface files
├── trained_models/                   # Saved .joblib model files
└── README.md                         # Project overview
```
# Clone the repository
https://github.com/Zain-ul-abdin0/Depression-Audio-AI-Analysis
- cd Depression-Audio-AI-Analysis
- python -m venv venv
- source venv/bin/activate     # For Linux/Mac
- venv\Scripts\activate        # For Windows

## Training & Evaluation
- python main.py
- Loading extracted features
- Model training and validation
- Saving trained models to trained_models/
## Web Interface (Deployment)
- python app.py
- http://localhost:5000
## Results
- Model evaluation metrics and visualizations are stored in the Results/ directory.
- This includes accuracy, confusion matrices, ROC curves, etc.

## Author
- Zain ul Abdin
- Department of Computer Science
- Supervisor: Dr. Roman Obermaisser
- Co-Supervisor: Ugonna Oleh
- Zain.Abdin@student.uni-siegen.de
## ⚠️ Important Disclaimer
-  This project is for research and educational purposes. It is a tool for early detection and monitoring, not a clinical diagnostic device. Always consult a healthcare professional for mental health concerns.

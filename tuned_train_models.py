
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score, average_precision_score
)
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.utils.class_weight import compute_class_weight
from joblib import dump

MODEL_SAVE_PATH = "trained_models"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def train_models(X, y, feature_names):
    if y is None:
        print("No labels available for training.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))

    results = {}

    cv_folds = 3 if len(y_train) < 60 else 5
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # Random Forest
    print("\n🔍 Tuning Random Forest…")
    rf = RandomForestClassifier(class_weight=class_weight_dict, random_state=42)
    rf_grid = {'n_estimators': [100, 300], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]}
    rf_search = GridSearchCV(rf, rf_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    rf_search.fit(X_train, y_train)
    rf_best = rf_search.best_estimator_

    print("✅ Best RF params:", rf_search.best_params_)
    rf_pred = rf_best.predict(X_test)
    rf_proba = rf_best.predict_proba(X_test)[:, 1]
    results["Random Forest"] = {
        "model": rf_best,
        "accuracy": accuracy_score(y_test, rf_pred),
        "roc_auc": roc_auc_score(y_test, rf_proba),
        "avg_precision": average_precision_score(y_test, rf_proba),
        "report": classification_report(y_test, rf_pred),
        "confusion_matrix": confusion_matrix(y_test, rf_pred),
        "feature_importances": list(zip(feature_names, rf_best.feature_importances_))
    }
    plt.figure()
    sns.heatmap(results["Random Forest"]["confusion_matrix"], annot=True, fmt='d', cmap='Blues')
    plt.title("Random Forest Confusion Matrix")
    plt.tight_layout()
    plt.savefig("random_forest_confusion_matrix.png")
    plt.show()

    # SVM
    print("\n🔍 Tuning SVM…")
    svm = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
    svm_grid = {'C': [0.01, 0.1, 1], 'gamma': ['scale', 'auto']}
    svm_search = GridSearchCV(svm, svm_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    svm_search.fit(X_train, y_train)
    svm_best = svm_search.best_estimator_
    print("✅ Best SVM params:", svm_search.best_params_)
    calibrated_svm = CalibratedClassifierCV(svm_best, method='sigmoid', cv=cv)
    calibrated_svm.fit(X_train, y_train)
    svm_pred = calibrated_svm.predict(X_test)
    svm_proba = calibrated_svm.predict_proba(X_test)[:, 1]
    results["SVM"] = {
        "model": calibrated_svm,
        "accuracy": accuracy_score(y_test, svm_pred),
        "roc_auc": roc_auc_score(y_test, svm_proba),
        "avg_precision": average_precision_score(y_test, svm_proba),
        "report": classification_report(y_test, svm_pred),
        "confusion_matrix": confusion_matrix(y_test, svm_pred),
        "feature_importances": None
    }
    plt.figure()
    sns.heatmap(results["SVM"]["confusion_matrix"], annot=True, fmt='d', cmap='Blues')
    plt.title("SVM Confusion Matrix")
    plt.tight_layout()
    plt.savefig("svm_confusion_matrix.png")
    plt.show()

    # XGBoost
    print("\n🔍 Tuning XGBoost…")
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        scale_pos_weight=sum(y_train == 0) / sum(y_train == 1),
        random_state=42
    )
    xgb_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.01, 0.05],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    xgb_search = GridSearchCV(xgb_model, xgb_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    xgb_search.fit(X_train, y_train)
    xgb_best = xgb_search.best_estimator_
    print("✅ Best XGBoost params:", xgb_search.best_params_)
    xgb_pred = xgb_best.predict(X_test)
    xgb_proba = xgb_best.predict_proba(X_test)[:, 1]
    results["XGBoost"] = {
        "model": xgb_best,
        "accuracy": accuracy_score(y_test, xgb_pred),
        "roc_auc": roc_auc_score(y_test, xgb_proba),
        "avg_precision": average_precision_score(y_test, xgb_proba),
        "report": classification_report(y_test, xgb_pred),
        "confusion_matrix": confusion_matrix(y_test, xgb_pred),
        "feature_importances": list(zip(feature_names, xgb_best.feature_importances_))
    }
    plt.figure()
    sns.heatmap(results["XGBoost"]["confusion_matrix"], annot=True, fmt='d', cmap='Blues')
    plt.title("XGBoost Confusion Matrix")
    plt.tight_layout()
    plt.savefig("xgboost_confusion_matrix.png")
    plt.show()

    # Ensemble
    print("\n✅ Building ensemble…")
    successful = [(k, v["model"]) for k, v in results.items()]
    ensemble = VotingClassifier(estimators=successful, voting="soft")
    ensemble.fit(X_train, y_train)
    ensemble_pred = ensemble.predict(X_test)
    results["Ensemble"] = {
        "model": ensemble,
        "accuracy": accuracy_score(y_test, ensemble_pred),
        "confusion_matrix": confusion_matrix(y_test, ensemble_pred)
    }
    plt.figure()
    sns.heatmap(results["Ensemble"]["confusion_matrix"], annot=True, fmt='d', cmap='Blues')
    plt.title("Ensemble Confusion Matrix")
    plt.tight_layout()
    plt.savefig("ensemble_confusion_matrix.png")
    plt.show()

    return results

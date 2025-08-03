import joblib

# Load the features
features = joblib.load('trained_models/overall_selected_features.joblib')

# Check the number of features
if isinstance(features, list):
    print(f"Number of selected features: {len(features)}")
    print("Feature names:", features)
elif hasattr(features, 'shape'):
    if len(features.shape) == 1:
        print(f"Number of selected features: {features.shape[0]}")
    else:
        print(f"Feature matrix shape: {features.shape}")  # (n_samples, n_features)
else:
    print("Features object type:", type(features))
    print("Contents:", features)
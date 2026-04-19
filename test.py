import joblib

try:
    model = joblib.load('knn_model_pipeline.joblib')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
import joblib
import numpy as np
import os

class ClassificationAgent:
    """
    Loads a trained ML model to classify feature vectors into Yoga Pose categories.
    """
    def __init__(self, model_path='model/model.pkl'):
        self.model_path = model_path
        self.model = None
        self.classes_ = None
        
        if os.path.exists(self.model_path):
            self.load_model()
            
    def load_model(self):
        try:
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.classes_ = model_data.get('classes', [])
        except Exception as e:
            print(f"Error loading model: {e}")
            
    def predict(self, feature_vector):
        """
        Returns the predicted class and the predicted probability (confidence) score.
        """
        if self.model is None:
            return "No model found", 0.0
            
        # Reshape for single sample prediction
        X = np.array([feature_vector])
        
        prediction = self.model.predict(X)[0]
        
        # Calculate confidence using predict_proba if available (RandomForest/SVM with probability=True)
        confidence = 0.0
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X)[0]
            confidence = max(probabilities) * 100
        
        return prediction, confidence

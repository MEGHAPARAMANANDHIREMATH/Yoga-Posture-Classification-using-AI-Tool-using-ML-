import numpy as np

class FeatureExtractionAgent:
    """
    Extracts numerical features (x, y, z, visibility) from MediaPipe landmarks
    to serve as input to the Machine Learning model.
    """
    def __init__(self):
        pass

    def extract_features(self, detection_result):
        """
        Takes MediaPipe Tasks vision result and returns a flattened list of features.
        Returns None if no pose is found.
        """
        if not detection_result or not detection_result.pose_landmarks:
            return None
            
        landmarks = []
        # Take the first detected pose
        pose_landmarks = detection_result.pose_landmarks[0]
        
        for landmark in pose_landmarks:
            landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
            
        # Flatten the list of lists into a 1D array
        feature_vector = np.array(landmarks).flatten().tolist()
        return feature_vector

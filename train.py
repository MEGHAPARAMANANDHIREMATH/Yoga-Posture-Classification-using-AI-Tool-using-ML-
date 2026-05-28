import os
import cv2
import numpy as np
from PIL import Image
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from agents.input_agent import InputAgent
from agents.pose_detection_agent import PoseDetectionAgent
from agents.feature_extraction_agent import FeatureExtractionAgent

def prepare_data(dataset_path):
    """
    Reads images from dataset_path, uses the agents to extract features, 
    and returns X (features) and y (labels).
    """
    input_agent = InputAgent()
    pose_agent = PoseDetectionAgent()
    feature_agent = FeatureExtractionAgent()
    
    X = []
    y = []
    
    if not os.path.exists(dataset_path):
        print(f"Warning: Dataset path {dataset_path} does not exist.")
        return X, y

    # Folder structure is dataset/split/ClassName/image.jpg
    for pose_class in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, pose_class)
        if not os.path.isdir(class_dir):
            continue
            
        print(f"Processing class: {pose_class} in {dataset_path}")
        
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            try:
                # Read image using OpenCV
                image = cv2.imread(image_path)
                if image is None:
                    continue
                    
                # Convert BGR to RGB as Mediapipe uses RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Standardize input
                image = input_agent.process_image(image)
                
                # Detect pose and extract features
                pose_results, _ = pose_agent.detect_pose(image)
                features = feature_agent.extract_features(pose_results)
                
                if features is not None:
                    X.append(features)
                    y.append(pose_class)
                    
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                
    return X, y

if __name__ == "__main__":
    print("Starting ML Training Pipeline...")
    
    # 1. Prepare Traing Data
    print("Loading Training Data...")
    X_train, y_train = prepare_data(os.path.join("dataset", "train"))
    
    # 2. Prepare Testing Data
    print("Loading Testing Data...")
    X_test, y_test = prepare_data(os.path.join("dataset", "test"))
    
    # Check if there is data
    if len(X_train) == 0:
        print("No training data found. Please place images in the dataset/train folder grouped by class.")
        exit(1)
        
    # If no test data provided, split train set
    if len(X_test) == 0:
        print("No test data found in dataset/test, splitting training data for validation.")
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # 3. Train Model
    print(f"Training Model on {len(X_train)} samples...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # 4. Evaluate Model
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Evaluation Accuracy on Test Set: {acc * 100:.2f}%")
    
    # 5. Save Model
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "model.pkl")
    # Save the model and the known classes
    joblib.dump({"model": clf, "classes": clf.classes_}, model_path)
    
    print(f"Training successful! Model saved to {model_path}")

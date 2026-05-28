# AI Agent-Based Yoga Posture Classification System

This is an end-to-end Machine Learning project that classifies yoga poses using a modular, agent-based architecture.
It can classify yoga poses from uploaded images and real-time webcam input using MediaPipe for pose detection and a Random Forest Classifier for classification.

## Supported Poses (Classes)
- Tree Pose
- Warrior Pose
- Cobra Pose
- Downward Dog
- T Pose

## Setup Instructions

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Organize the dataset:
   Place your images in the `dataset/train/` and `dataset/test/` folders inside the respective class subfolders:
   ```
   dataset/
      train/
         Tree/
         Warrior/
         Cobra/
         DownwardDog/
         TPose/
      test/
         Tree/
         Warrior/
         Cobra/
         DownwardDog/
         TPose/
   ```

3. Train the model:
   Run the training script to extract features from images, train a Random Forest classifier, and save the model:
   ```bash
   python train.py
   ```
   The model will be saved as `model/model.pkl`.

4. Run the UI:
   Launch the Gradio Web App for evaluation and real-time inference:
   ```bash
   python app.py
   ```

## Agent Architecture
- **InputAgent:** Handles loading webcam frames or uploaded images.
- **PoseDetectionAgent:** Extracts Body landmarks via MediaPipe Pose model.
- **FeatureExtractionAgent:** Converts landmarks to flattened structured feature vectors.
- **ClassificationAgent:** Loads `model/model.pkl` and predicts the yoga pose.
- **FeedbackAgent:** Generates helpful feedback messages based on the recognized pose.

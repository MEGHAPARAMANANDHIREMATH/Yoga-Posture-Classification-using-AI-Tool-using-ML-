import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class PoseDetectionAgent:
    """
    Given an image, uses MediaPipe Tasks API to extract body landmarks.
    Compatible with Python 3.13.
    """
    def __init__(self, model_asset_path='model/pose_landmarker_lite.task'):
        base_options = python.BaseOptions(model_asset_path=model_asset_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            num_poses=1)
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def detect_pose(self, image):
        """
        Processes RGB image and returns results object with landmarks.
        Draws landmarks on the image for visualization using cv2.
        """
        # Convert numpy array to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        
        # Detect pose
        detection_result = self.detector.detect(mp_image)
        
        annotated_image = image.copy()
        
        # Manually draw landmarks using cv2
        if detection_result.pose_landmarks:
            for pose_landmarks in detection_result.pose_landmarks:
                for landmark in pose_landmarks:
                    h, w, c = annotated_image.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    
                    if landmark.visibility > 0.5:
                        cv2.circle(annotated_image, (cx, cy), 4, (245, 117, 66), -1)
                        
        return detection_result, annotated_image

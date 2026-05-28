class FeedbackAgent:
    """
    Provides feedback based on the predicted pose and current state.
    """
    def __init__(self):
        self.feedback_map = {
            "tree": "Great Tree pose! Keep your standing leg straight and focus on a point in front of you.",
            "warrior2": "Strong Warrior II! Make sure your front knee is directly above your ankle.",
            "goddess": "Nice Goddess pose! Keep your chest lifted and knees tracking over toes.",
            "downdog": "Good Downward Dog! Press firmly into your hands and send your hips high.",
            "plank": "Strong Plank! Keep your core engaged and body in a straight line.",
            "NoPose": "No pose detected. Please step back into the camera frame.",
            "LowConfidence": "I'm unsure about the pose. Try to make your posture more distinct."
        }

    def generate_feedback(self, pose_name, confidence):
        """
        Returns a feedback string given the pose and confidence.
        """
        if pose_name == "No model found":
            return "Model not trained yet! Please run the training script."
            
        if pose_name is None:
            return self.feedback_map["NoPose"]
            
        if confidence < 40.0:
            return self.feedback_map["LowConfidence"]
            
        # Get specific feedback or a generic one
        return self.feedback_map.get(pose_name, f"{pose_name} detected! Great job.")

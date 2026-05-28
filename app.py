import gradio as gr
import cv2
import numpy as np

# Agent Imports
from agents.input_agent import InputAgent
from agents.pose_detection_agent import PoseDetectionAgent
from agents.feature_extraction_agent import FeatureExtractionAgent
from agents.classification_agent import ClassificationAgent
from agents.feedback_agent import FeedbackAgent

# Initialize agents once globally
input_agent = InputAgent()
pose_agent = PoseDetectionAgent()
feature_agent = FeatureExtractionAgent()
classification_agent = ClassificationAgent()
feedback_agent = FeedbackAgent()

def process_frame(image):
    """
    Routinely processes a single frame/image through the ML agent pipeline.
    """
    if image is None:
        return None, "No Image", "0.00%", "Please upload an image or turn on the webcam.", gr.update(interactive=False)
    
    try:
        # 1. Standardize Input
        image_rgb = input_agent.process_image(image)
        
        # 2. Detect Pose Landmarks and Annotate Diagram
        pose_results, annotated_image = pose_agent.detect_pose(image_rgb)
        
        # 3. Extract Features
        features = feature_agent.extract_features(pose_results)
        
        if features is None:
            return annotated_image, "No Pose Detected", "0.00%", "Make sure your full body is visible in the frame.", gr.update(interactive=True)
            
        # 4. Classification
        predicted_pose, confidence = classification_agent.predict(features)
        
        # 5. Generate Feedback
        feedback = feedback_agent.generate_feedback(predicted_pose, confidence)
        
        return annotated_image, predicted_pose, f"{confidence:.2f}%", feedback, gr.update(interactive=True)
        
    except Exception as e:
        return image, "Error", "0.00%", str(e), gr.update(interactive=True)


# === Gradio UI ===
custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}
.gr-box, .gr-panel, .gr-block {
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
    border-radius: 16px !important;
}
h1 {
    text-align: center;
    font-weight: 800 !important;
    text-shadow: 0 0 20px rgba(0, 229, 255, 0.4);
    background: -webkit-linear-gradient(45deg, #00e5ff, #ffffff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    padding-bottom: 10px;
}
.gr-button-primary {
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(0, 229, 255, 0.3) !important;
}
.gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(0, 229, 255, 0.5) !important;
}
/* Scanning Animation CSS */
.scanner-container {
    position: relative;
    overflow: hidden;
    border-radius: 12px;
}
.scan-line {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 4px;
    background: #00e5ff;
    box-shadow: 0 0 10px #00e5ff, 0 0 20px #00e5ff;
    animation: scan 2.5s cubic-bezier(0.4, 0.0, 0.2, 1) infinite;
    z-index: 100;
    opacity: 0.8;
    pointer-events: none;
}
@keyframes scan {
    0% { top: 0%; opacity: 0; }
    10% { opacity: 1; }
    90% { opacity: 1; }
    100% { top: 100%; opacity: 0; }
}
/* Landing Page Specific */
.landing-hero {
    text-align: center;
    padding: 4rem 2rem 2rem 2rem;
}
.landing-hero h1 {
    font-size: 3.5rem !important;
    margin-bottom: 1.5rem !important;
    line-height: 1.2;
}
.landing-hero p {
    font-size: 1.2rem;
    color: rgba(255,255,255,0.9) !important;
    max-width: 700px;
    margin: 0 auto 2rem auto;
    line-height: 1.6;
}
.start-btn {
    font-size: 1.5rem !important;
    padding: 1.5rem 3rem !important;
    border-radius: 50px !important;
    margin-top: 1rem !important;
    margin-bottom: 3rem !important;
}
.features-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 2rem;
    margin-top: 2rem;
    padding: 0 2rem 4rem 2rem;
}
.feature-card {
    background: rgba(255, 255, 255, 0.05);
    padding: 2rem;
    border-radius: 16px;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.1);
    transition: transform 0.3s;
}
.feature-card:hover {
    transform: translateY(-5px);
    background: rgba(255, 255, 255, 0.08);
}
.feature-icon {
    font-size: 2.5rem;
    margin-bottom: 1rem;
}
"""

modern_theme = gr.themes.Base(
    primary_hue="cyan",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Outfit"), "sans-serif"]
).set(
    body_background_fill="linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%)",
    body_background_fill_dark="linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%)",
    body_text_color="white",
    body_text_color_dark="white",
    background_fill_primary="rgba(255, 255, 255, 0.05)",
    background_fill_primary_dark="rgba(255, 255, 255, 0.05)",
    background_fill_secondary="rgba(0, 0, 0, 0.2)",
    background_fill_secondary_dark="rgba(0, 0, 0, 0.2)",
    border_color_primary="rgba(255, 255, 255, 0.1)",
    border_color_primary_dark="rgba(255, 255, 255, 0.1)",
    block_background_fill="rgba(255, 255, 255, 0.05)",
    block_background_fill_dark="rgba(255, 255, 255, 0.05)",
    block_border_width="1px",
    block_border_color="rgba(255, 255, 255, 0.1)",
    block_border_color_dark="rgba(255, 255, 255, 0.1)",
    input_background_fill="rgba(0, 0, 0, 0.2)",
    input_background_fill_dark="rgba(0, 0, 0, 0.2)",
    button_primary_background_fill="linear-gradient(45deg, #00e5ff, #0076ff)",
    button_primary_background_fill_dark="linear-gradient(45deg, #00e5ff, #0076ff)",
    button_primary_text_color="white",
    button_secondary_background_fill="rgba(255, 255, 255, 0.1)",
    button_secondary_background_fill_dark="rgba(255, 255, 255, 0.1)",
    button_secondary_text_color="white",
)

# Creating modern interface
with gr.Blocks(title="AI Yoga Pose Classification", theme=modern_theme, css=custom_css) as app:
    
    # --- Landing Page ---
    with gr.Column(visible=True) as landing_page:
        gr.HTML('''
        <div class="landing-hero">
            <h1>Elevate Your Yoga Practice with AI</h1>
            <p>Our intelligent multi-agent system acts as your personal virtual instructor. It analyzes your form in real-time, recognizes standard poses, and provides tailored, actionable feedback to ensure you practice safely and effectively.</p>
        </div>
        ''')
        
        with gr.Row():
            gr.Column(scale=1)
            start_btn = gr.Button("Get Started 🚀", variant="primary", elem_classes=["start-btn"])
            gr.Column(scale=1)
            
        gr.HTML('''
        <div class="features-grid">
            <div class="feature-card">
                <div class="feature-icon">🎯</div>
                <h3>Real-Time Tracking</h3>
                <p>33-point body landmark detection running instantly on your camera feed.</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🧠</div>
                <h3>Smart Analysis</h3>
                <p>Calculates precise joint angles and body structure to classify your pose accurately.</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">💬</div>
                <h3>Instant Feedback</h3>
                <p>Receive corrective advice to improve your alignment and prevent injuries.</p>
            </div>
        </div>
        ''')
            
    # --- Scanner Page ---
    with gr.Column(visible=False) as scanner_page:
        with gr.Row():
            back_btn = gr.Button("⬅️ Back to Home", variant="secondary", size="sm")
            gr.Markdown("## 🧘‍♀️ Yoga Pose Scanner")
            
        with gr.Tabs():
            
            # Image Upload Tab
            with gr.TabItem("🖼️ Image Upload"):
                with gr.Row():
                    with gr.Column(scale=1, elem_classes=["scanner-container"]):
                        gr.HTML('<div class="scan-line"></div>')
                        image_input = gr.Image(type="numpy", label="Upload Yoga Image")
                        img_analyze_btn = gr.Button("Analyze Pose", variant="primary")
                    with gr.Column(scale=1):
                        image_output = gr.Image(label="Pose Detected")
                        with gr.Row():
                            image_pose = gr.Textbox(label="Predicted Pose")
                            image_conf = gr.Textbox(label="Confidence")
                        image_feed = gr.Textbox(label="AI Feedback", lines=3)
                        img_scan_another_btn = gr.Button("Scan Another", variant="secondary", interactive=False)
                        
                img_analyze_btn.click(
                    fn=process_frame,
                    inputs=image_input,
                    outputs=[image_output, image_pose, image_conf, image_feed, img_scan_another_btn]
                )
                
                img_scan_another_btn.click(
                    fn=lambda: (None, None, "", "", "", gr.update(interactive=False)),
                    inputs=[],
                    outputs=[image_input, image_output, image_pose, image_conf, image_feed, img_scan_another_btn]
                )
                
            # Webcam Tab
            with gr.TabItem("📷 Live Webcam"):
                gr.Markdown("### Real-Time Yoga Assistant")
                gr.Markdown("Ensure you have granted camera permissions. The processing runs directly on your machine!")
                
                with gr.Row():
                    with gr.Column(scale=1, elem_classes=["scanner-container"]):
                        gr.HTML('<div class="scan-line"></div>')
                        webcam_input = gr.Image(sources=["webcam"], type="numpy", streaming=True, label="Webcam Feed")
                    with gr.Column(scale=1):
                        webcam_output = gr.Image(label="Pose Processed")
                        with gr.Row():
                            webcam_pose = gr.Textbox(label="Predicted Pose")
                            webcam_conf = gr.Textbox(label="Confidence")
                        webcam_feed = gr.Textbox(label="AI Feedback", lines=3)
                        webcam_scan_another_btn = gr.Button("Scan Another", variant="secondary", interactive=False)
                        
                webcam_input.stream(
                    fn=process_frame,
                    inputs=webcam_input,
                    outputs=[webcam_output, webcam_pose, webcam_conf, webcam_feed, webcam_scan_another_btn]
                )
                
                webcam_scan_another_btn.click(
                    fn=lambda: (None, None, "", "", "", gr.update(interactive=False)),
                    inputs=[],
                    outputs=[webcam_input, webcam_output, webcam_pose, webcam_conf, webcam_feed, webcam_scan_another_btn]
                )
                
        gr.Markdown("---")
        gr.Markdown("**Note:** Model accuracy depends heavily on your training dataset and room lighting.")

    # Navigation Events
    start_btn.click(lambda: (gr.update(visible=False), gr.update(visible=True)), None, [landing_page, scanner_page])
    back_btn.click(lambda: (gr.update(visible=True), gr.update(visible=False)), None, [landing_page, scanner_page])

if __name__ == "__main__":
    app.queue()
    app.launch()

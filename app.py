import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
import base64
from io import BytesIO

from flask import Flask, request, jsonify
from flask_cors import CORS

# Grad-CAM libraries (ensure you have pytorch-grad-cam installed)
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from your frontend

# --- General Configuration ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# =============================================================================
# CHEST X-RAY CONFIGURATION
# =============================================================================
CHEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_covid_model.pth") # Using the better COVID model
CHEST_LABELS = ['Normal', 'COVID', 'Lung_Opacity', 'Viral_Pneumonia']

chest_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =============================================================================
# BRAIN TUMOR (MRI) CONFIGURATION
# =============================================================================
BRAIN_MODEL_PATH = os.path.join(MODEL_DIR, 'densenet_brain_tumor_v3.pth')
BRAIN_LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']

brain_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3), # Brain model was trained on grayscale
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =============================================================================
# MODEL LOADING FUNCTIONS
# =============================================================================
def load_chest_model():
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(1024, len(CHEST_LABELS))
    # Load model trained for COVID/Pneumonia classification
    checkpoint = torch.load(CHEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

def load_brain_model():
    model = models.densenet121(weights=None)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, len(BRAIN_LABELS))
    model.load_state_dict(torch.load(BRAIN_MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# =============================================================================
# PREDICTION & GRAD-CAM FUNCTIONS
# =============================================================================

# --- CHEST X-RAY ---
def predict_chest(image: Image.Image, model):
    image_rgb = image.convert("RGB")
    image_tensor = chest_transform(image_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    class_idx = np.argmax(probs)
    return CHEST_LABELS[class_idx], float(probs[class_idx]), probs, image_tensor

def generate_chest_gradcam(model, image_tensor, target_layer, class_idx, orig_image: Image.Image):
    orig_np = np.array(orig_image.convert("RGB").resize((224, 224))).astype(np.float32) / 255.0
    cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=image_tensor, targets=[ClassifierOutputTarget(class_idx)])[0]
    cam_image = show_cam_on_image(orig_np, grayscale_cam, use_rgb=True)
    pil_image = Image.fromarray(cam_image)
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# --- BRAIN MRI ---
def predict_brain(image: Image.Image, model):
    image_rgb = image.convert("RGB") # Even if transform makes it grayscale, start with RGB
    image_tensor = brain_transform(image_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
    predicted_idx = np.argmax(probs)
    predicted_class = BRAIN_LABELS[predicted_idx]
    confidence_score = float(probs[predicted_idx])
    return predicted_class, confidence_score, probs, image_tensor

def generate_brain_gradcam(model, image_tensor, target_layers, class_idx, orig_image: Image.Image):
    rgb_img_float = np.array(orig_image.convert("RGB").resize((224, 224))) / 255.0
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=image_tensor, targets=[ClassifierOutputTarget(class_idx)])[0, :]
    cam_image = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)
    pil_image = Image.fromarray(cam_image)
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# =============================================================================
# LOAD ALL MODELS ON STARTUP
# =============================================================================
print(" * Loading Chest X-Ray model...")
chest_model = load_chest_model()
chest_target_layer = chest_model.features.denseblock4.denselayer16.conv2
print("✅ Chest X-Ray model loaded.")

print(" * Loading Brain Tumor model...")
brain_model = load_brain_model()
brain_target_layer = [brain_model.features] # Grad-CAM for this model targets the whole feature block
print("✅ Brain Tumor model loaded.")


# =============================================================================
# API ENDPOINTS
# =============================================================================

def process_request(request, model_name):
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    try:
        image = Image.open(file.stream)

        if model_name == 'chest':
            label, prob, all_probs, image_tensor = predict_chest(image, chest_model)
            class_idx = CHEST_LABELS.index(label)
            heatmap_base64 = generate_chest_gradcam(chest_model, image_tensor, chest_target_layer, class_idx, image)
            labels_list = CHEST_LABELS
        
        elif model_name == 'brain':
            label, prob, all_probs, image_tensor = predict_brain(image, brain_model)
            class_idx = BRAIN_LABELS.index(label)
            heatmap_base64 = generate_brain_gradcam(brain_model, image_tensor, brain_target_layer, class_idx, image)
            labels_list = BRAIN_LABELS

        result = {
            "label": label,
            "confidence": round(prob, 4),
            "probabilities": [
                {"label": labels_list[i], "value": round(float(p), 4)}
                for i, p in enumerate(all_probs)
            ],
            "heatmap": heatmap_base64
        }
        return jsonify(result)

    except Exception as e:
        print(f"❌ An error occurred during {model_name} prediction: {e}")
        return jsonify({"error": f"An error occurred processing the image for {model_name}."}), 500

@app.route("/api/chest", methods=["POST"])
def predict_chest_endpoint():
    """Endpoint for Chest X-Ray analysis."""
    return process_request(request, 'chest')

@app.route("/api/brain", methods=["POST"])
def predict_brain_endpoint():
    """Endpoint for Brain Tumor (MRI) analysis."""
    return process_request(request, 'brain')

# =============================================================================
# RUN THE APP
# =============================================================================
if __name__ == "__main__":
    # Use 0.0.0.0 to make it accessible from your Next.js app running in a container or on another device
    app.run(host="0.0.0.0", port=5000, debug=True)
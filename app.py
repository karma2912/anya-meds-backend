import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image
import os
import numpy as np
import base64
from io import BytesIO
import traceback

# Imports for Gemini
import google.generativeai as genai
from dotenv import load_dotenv

# Imports for the Skin model
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from flask import Flask, request, jsonify
from flask_cors import CORS

# Grad-CAM libraries
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- Initialize Flask App ---
app = Flask(__name__)
origins = [
    "http://localhost:3000",
    r"https?://.*\.netlify\.app" # A regular expression for all netlify.app subdomains
]
CORS(app, resources={r"/api/*": {"origins": origins}})

# --- General Configuration ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# --- Configure Gemini API ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("⚠️ WARNING: GEMINI_API_KEY environment variable not set. AI Summaries will be disabled.")
else:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini API configured successfully.")

# =============================================================================
# SKIN LESION MODEL DEFINITION (Required for loading)
# =============================================================================
class EnhancedModel(nn.Module):
    def __init__(self, model_name, num_classes, dropout_rate=0.3):
        super(EnhancedModel, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='')
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.BatchNorm1d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.25),
            nn.Linear(feature_dim // 2, num_classes)
        )
    def forward(self, x):
        features = self.backbone(x)
        avg_pool = self.global_avg_pool(features).flatten(1)
        max_pool = self.global_max_pool(features).flatten(1)
        combined = torch.cat([avg_pool, max_pool], dim=1)
        return self.classifier(combined)

# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================
# CHEST X-RAY
CHEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_covid_model.pth")
CHEST_LABELS = ['Normal', 'COVID', 'Lung_Opacity', 'Viral_Pneumonia']
chest_transform = A.Compose([A.Resize(224, 224), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()])

# BRAIN TUMOR (MRI)
BRAIN_MODEL_PATH = os.path.join(MODEL_DIR, 'densenet_brain_tumor_v3.pth')
BRAIN_LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']
brain_transform = A.Compose([A.Resize(224, 224), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()])

# SKIN LESION
SKIN_MODEL_PATH = os.path.join(MODEL_DIR, 'skin-model.pth')
SKIN_MODEL_NAME = 'tf_efficientnet_b3.ns_jft_in1k' # Using updated name to avoid warning
SKIN_IMG_SIZE = 256
SKIN_NUM_CLASSES = 7
SKIN_DROPOUT_RATE = 0.4
SKIN_LABELS_SHORT = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
SKIN_LABELS_FULL = {'akiec': 'Actinic Keratoses', 'bcc': 'Basal Cell Carcinoma', 'bkl': 'Benign Keratosis-like Lesions', 'df': 'Dermatofibroma', 'mel': 'Melanoma', 'nv': 'Melanocytic Nevi', 'vasc': 'Vascular Lesions'}
skin_transform = A.Compose([A.Resize(SKIN_IMG_SIZE, SKIN_IMG_SIZE), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()])

# =============================================================================
# LAZY LOADING SETUP
# =============================================================================
print("🚀 AnYa Med AI Server starting up...")
# Initialize models as None. They will be loaded on the first request to their endpoint.
chest_model, brain_model, skin_model = None, None, None

# Define target layers (attributes are light and don't load models)
chest_target_layer = models.densenet121().features.denseblock4.denselayer16.conv2
brain_target_layer = models.densenet121().features.denseblock4.denselayer16.conv2

print("🎉 Server is ready. Models will be loaded on demand.")

# =============================================================================
# MODEL LOADING & PREDICTION FUNCTIONS
# =============================================================================
def load_chest_model():
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(1024, len(CHEST_LABELS))
    checkpoint = torch.load(CHEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def load_brain_model():
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, len(BRAIN_LABELS))
    model.load_state_dict(torch.load(BRAIN_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_skin_model():
    model = EnhancedModel(model_name=SKIN_MODEL_NAME, num_classes=SKIN_NUM_CLASSES, dropout_rate=SKIN_DROPOUT_RATE)
    model.load_state_dict(torch.load(SKIN_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

def process_image(image: Image.Image, transform):
    image_rgb = image.convert("RGB")
    image_np = np.array(image_rgb)
    transformed = transform(image=image_np)
    return transformed['image'].unsqueeze(0).to(device)

def generate_gradcam(model, target_layer, image_tensor, orig_image: Image.Image, class_idx, img_size=224, cam_method=GradCAM):
    orig_np = np.array(orig_image.convert("RGB").resize((img_size, img_size))).astype(np.float32) / 255.0
    cam = cam_method(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(class_idx)]
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0, :]
    cam_image = show_cam_on_image(orig_np, grayscale_cam, use_rgb=True)
    pil_image = Image.fromarray(cam_image)
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def predict_model(image_tensor, model, labels):
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
    predicted_idx = np.argmax(probs)
    return labels[predicted_idx], float(probs[predicted_idx]), probs

def generate_ai_summary(model_name, diagnosis, confidence):
    if not GEMINI_API_KEY:
        return "AI Summary feature is disabled. Please configure your Gemini API Key."
    prompts = {
        "chest": f"As a medical scribe summarizing an AI analysis of a chest X-ray, write a brief, objective clinical summary in 1-2 sentences. The model's primary finding is '{diagnosis}' with a confidence of {confidence*100:.1f}%.",
        "brain": f"As a medical scribe summarizing an AI analysis of a brain MRI, write a brief, objective clinical summary in 1-2 sentences. The model's primary finding is a tumor classified as '{diagnosis}' with a confidence of {confidence*100:.1f}%.",
        "skin": f"As a medical scribe summarizing an AI analysis of a dermoscopic image, write a brief, objective clinical summary in 1-2 sentences. The AI identified the lesion as '{diagnosis}' with a confidence of {confidence*100:.1f}%."
    }
    prompt = prompts.get(model_name, f"Provide a brief summary for the medical finding: {diagnosis}")
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"❌ Gemini API Error: {e}")
        return "Could not generate AI summary due to an API error."

# =============================================================================
# API ENDPOINTS
# =============================================================================
def process_request(request, model_name):
    global chest_model, brain_model, skin_model
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    try:
        image = Image.open(file.stream).convert("RGB")
        if model_name == 'chest':
            if chest_model is None:
                print(" * Loading Chest X-Ray model for the first time...")
                chest_model = load_chest_model()
                print("✅ Chest X-Ray model loaded.")
            image_tensor = process_image(image, chest_transform)
            label, prob, all_probs = predict_model(image_tensor, chest_model, CHEST_LABELS)
            labels_list = CHEST_LABELS
            class_idx = labels_list.index(label)
            heatmap_base64 = generate_gradcam(chest_model, chest_target_layer, image_tensor, image, class_idx, img_size=224, cam_method=GradCAMPlusPlus)
        elif model_name == 'brain':
            if brain_model is None:
                print(" * Loading Brain Tumor model for the first time...")
                brain_model = load_brain_model()
                print("✅ Brain Tumor model loaded.")
            image_tensor = process_image(image, brain_transform)
            label, prob, all_probs = predict_model(image_tensor, brain_model, BRAIN_LABELS)
            labels_list = BRAIN_LABELS
            class_idx = labels_list.index(label)
            heatmap_base64 = generate_gradcam(brain_model, brain_target_layer, image_tensor, image, class_idx, img_size=224, cam_method=GradCAMPlusPlus)
        elif model_name == 'skin':
            if skin_model is None:
                print(" * Loading Skin Lesion model for the first time...")
                skin_model = load_skin_model()
                print("✅ Skin Lesion model loaded.")
            skin_target_layer = skin_model.backbone.conv_head
            image_tensor = process_image(image, skin_transform)
            label_short, prob, all_probs = predict_model(image_tensor, skin_model, SKIN_LABELS_SHORT)
            label = SKIN_LABELS_FULL[label_short]
            labels_list = [SKIN_LABELS_FULL[l] for l in SKIN_LABELS_SHORT]
            class_idx = SKIN_LABELS_SHORT.index(label_short)
            heatmap_base64 = generate_gradcam(skin_model, skin_target_layer, image_tensor, image, class_idx, img_size=SKIN_IMG_SIZE)
        else:
            return jsonify({"error": "Invalid model name specified"}), 400

        ai_summary = generate_ai_summary(model_name, label, prob)
        result = {
            "label": label,
            "confidence": round(prob, 4),
            "probabilities": [{"label": labels_list[i], "value": round(float(p), 4)} for i, p in enumerate(all_probs)],
            "heatmap": heatmap_base64,
            "ai_summary": ai_summary
        }
        return jsonify(result)
    except Exception as e:
        print(f"❌ An error occurred during {model_name} prediction: {e}")
        traceback.print_exc()
        return jsonify({"error": f"An error occurred processing the image for {model_name}."}), 500

@app.route("/api/chest", methods=["POST"])
def predict_chest_endpoint():
    return process_request(request, 'chest')

@app.route("/api/brain", methods=["POST"])
def predict_brain_endpoint():
    return process_request(request, 'brain')

@app.route("/api/skin", methods=["POST"])
def predict_skin_endpoint():
    return process_request(request, 'skin')

# =============================================================================
# RUN THE APP (For local development only)
# =============================================================================
if __name__ == "__main__":
    # This block is for local development.
    # A production server like Gunicorn or Waitress will import the 'app' object directly.
    app.run(host="0.0.0.0", port=5000, debug=False) # Debug mode is OFF for production readiness
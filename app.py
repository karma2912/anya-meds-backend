import torch
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
import cv2
import json
import base64
from io import BytesIO

from flask import Flask, request, jsonify
from flask_cors import CORS

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- Initialize Flask App ---
app = Flask(__name__)
# This is crucial for allowing your Next.js frontend to call this API
CORS(app) 

# --- Model & Path Configuration ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
NIH_MODEL_PATH = os.path.join(MODEL_DIR, "densenet121_xray_optimized.pth") 
COVID_MODEL_PATH = os.path.join(MODEL_DIR, "best_covid_model.pth") 

THRESHOLD = 0.4
USE_COVID_MODEL = True # Set to True to use the COVID-19 model

NIH_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'No Finding'
]
COVID_LABELS = ['Normal', 'COVID', 'Lung_Opacity', 'Viral_Pneumonia']

# --- Device and Transforms ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Model Loading Functions (Unchanged) ---
def load_nih_model():
    model = models.densenet121(weights=None)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(1024, 256),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(256, len(NIH_LABELS))
    )
    model.load_state_dict(torch.load(NIH_MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def load_covid_model():
    model = models.densenet121(weights=None)
    model.classifier = torch.nn.Linear(1024, 4)
    checkpoint = torch.load(COVID_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

# --- Prediction Functions (Refactored to accept PIL Image object) ---
def predict_nih(image: Image.Image, model):
    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad(): 
        outputs = model(image_tensor)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
    
    predictions = [(NIH_LABELS[i], float(prob)) for i, prob in enumerate(probs) if prob >= THRESHOLD]
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions, probs, image_tensor

def predict_covid(image: Image.Image, model):
    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    class_idx = np.argmax(probs)
    return COVID_LABELS[class_idx], float(probs[class_idx]), probs, image_tensor

# --- Grad-CAM Function (Refactored to accept PIL Image object) ---
def generate_gradcam_base64(model, image_tensor, target_layer, class_idx, orig_image: Image.Image):
    high_res_size = (224, 224) 
    orig_np = np.array(orig_image.convert("RGB").resize(high_res_size)).astype(np.float32) / 255.0

    cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=image_tensor, targets=[ClassifierOutputTarget(class_idx)])[0]
    
    cam_image = show_cam_on_image(orig_np, grayscale_cam, use_rgb=True)

    img_pil = Image.fromarray(cam_image)
    buffered = BytesIO()
    img_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

# --- Load Model on Startup ---
print(" * Loading model... please wait.")
if USE_COVID_MODEL:
    model = load_covid_model()
    target_layer = model.features.denseblock4.denselayer16.conv2 # Target layer for Grad-CAM
else:
    model = load_nih_model()
    target_layer = model.features.denseblock4.denselayer16.conv2 # Adjust if needed for NIH model
print(" * Model loaded.")


# --- API Endpoint ---
@app.route("/api/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Read the image in-memory
        image = Image.open(file.stream)

        if USE_COVID_MODEL:
            label, prob, all_probs, image_tensor = predict_covid(image, model)
            class_idx_for_gradcam = COVID_LABELS.index(label)
            heatmap_base64 = generate_gradcam_base64(model, image_tensor, target_layer, class_idx_for_gradcam, image)

            result = {
                "label": label,
                "confidence": round(prob, 4),
                "probabilities": [
                    {"label": COVID_LABELS[i], "value": round(float(p), 4)}
                    for i, p in enumerate(all_probs)
                ],
                "heatmap": heatmap_base64
            }
        else:
            predictions, all_probs, image_tensor = predict_nih(image, model)
            result = {
                "label": predictions[0][0] if predictions else "No Finding",
                "confidence": round(predictions[0][1], 4) if predictions else 0.0,
                "probabilities": [
                    {"label": NIH_LABELS[i], "value": round(float(prob), 4)}
                    for i, prob in enumerate(all_probs)
                ],
                "heatmap": None # Grad-CAM can be implemented for NIH if needed
            }

        return jsonify(result)

    except Exception as e:
        # Log the error for debugging
        print(f"An error occurred: {e}")
        return jsonify({"error": "An error occurred during prediction. Please check the image file."}), 500

# --- Run the App ---
if __name__ == "__main__":
    # Use 0.0.0.0 to make it accessible from other devices on the same network
    app.run(host="0.0.0.0", port=5000, debug=True)
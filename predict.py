import torch
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
import cv2
import json
import sys
import base64
from io import BytesIO

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
os.makedirs(MODEL_DIR, exist_ok=True)
NIH_MODEL_PATH = os.path.join(MODEL_DIR, "densenet121_xray_optimized.pth") 
COVID_MODEL_PATH = os.path.join(MODEL_DIR, "best_covid_model.pth") 

THRESHOLD = 0.4
USE_COVID_MODEL = True

NIH_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'No Finding'
]
COVID_LABELS = ['Normal', 'COVID', 'Lung_Opacity', 'Viral_Pneumonia']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_nih_model():
    model = models.densenet121(weights=None)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(1024, 256),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(256, len(NIH_LABELS))
    )
    state_dict = torch.load(NIH_MODEL_PATH, map_location=device)
    model_state_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
    model_state_dict.update(pretrained_dict)
    model.load_state_dict(model_state_dict)
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

def predict_nih(image_path, model):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad(): 
        outputs = model(image_tensor)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
    
    predictions = [(NIH_LABELS[i], float(prob)) for i, prob in enumerate(probs) if prob >= THRESHOLD]
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions, probs, image_tensor

def predict_covid(image_path, model):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    class_idx = np.argmax(probs)
    return COVID_LABELS[class_idx], float(probs[class_idx]), probs, image_tensor

def generate_gradcam_base64(model, image_tensor, target_layer, class_idx, orig_image_path):
    model.eval()
    image_tensor = image_tensor.to(device)

    high_res_size = (224, 224) 
    orig = Image.open(orig_image_path).convert("RGB").resize(high_res_size)
    orig_np = np.array(orig).astype(np.float32) / 255.0

    cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=image_tensor, targets=[ClassifierOutputTarget(class_idx)])[0]
    grayscale_cam = cv2.resize(grayscale_cam, high_res_size)
    cam_image = show_cam_on_image(orig_np, grayscale_cam, use_rgb=True)

    img_pil = Image.fromarray(cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
    buffered = BytesIO()
    img_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({ "error": "No image path provided as argument." }))
        sys.exit(1)

    IMAGE_PATH = sys.argv[1]

    if not os.path.exists(IMAGE_PATH):
        print(json.dumps({ "error": f"Image file not found at: {IMAGE_PATH}" }))
        sys.exit(1)

    try:
        if USE_COVID_MODEL:
            model = load_covid_model()
            label, prob, all_probs, image_tensor = predict_covid(IMAGE_PATH, model)
            target_layer = model.features.denseblock4.denselayer16.conv2
            class_idx_for_gradcam = COVID_LABELS.index(label) if label in COVID_LABELS else 0 
            heatmap_base64 = generate_gradcam_base64(model, image_tensor, target_layer, class_idx_for_gradcam, IMAGE_PATH)

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
            model = load_nih_model()
            predictions, all_probs, image_tensor = predict_nih(IMAGE_PATH, model)
            
            result = {
                "label": predictions[0][0] if predictions else "No Finding",
                "confidence": round(predictions[0][1], 4) if predictions else 0.0,
                "probabilities": [
                    {"label": NIH_LABELS[i], "value": round(float(prob), 4)}
                    for i, prob in enumerate(all_probs)
                ],
                "heatmap": None 
            }

        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({ "error": str(e) }))
        sys.exit(1)
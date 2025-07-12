import torch
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np

# Configuration
NIH_MODEL_PATH = "model/densenet121.pth"
COVID_MODEL_PATH = "best_covid_model.pth"
IMAGE_PATH = "test.png"
THRESHOLD = 0.4  # For NIH multi-label classification
USE_COVID_MODEL = True  # ✅ Change to False if you want to run NIH model

# Labels
NIH_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'No Finding'
]

COVID_LABELS = ['Normal', 'COVID', 'Lung_Opacity', 'Viral_Pneumonia']

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Common transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_nih_model():
    print("🔍 Loading NIH model...")
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
    print("✅ NIH model loaded!")
    return model

def load_covid_model():
    print("🦠 Loading COVID-19 model...")
    model = models.densenet121(weights=None)
    model.classifier = torch.nn.Linear(1024, 4)  # 4 classes
    checkpoint = torch.load(COVID_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("✅ COVID-19 model loaded!")
    return model

def predict_nih(image_path, model):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
    predictions = [(NIH_LABELS[i], float(prob)) for i, prob in enumerate(probs) if prob >= THRESHOLD]
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions, probs

def predict_covid(image_path, model):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    class_idx = np.argmax(probs)
    return COVID_LABELS[class_idx], float(probs[class_idx]), probs

if __name__ == "__main__":
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"❌ Image not found: {IMAGE_PATH}")

    print("\n🩻 Running prediction...\n")

    if USE_COVID_MODEL:
        model = load_covid_model()
        label, prob, all_probs = predict_covid(IMAGE_PATH, model)
        print(f"🦠 Predicted condition: {label} ({prob:.1%} confidence)")
        print("\n🔍 Class probabilities:")
        for i, p in enumerate(all_probs):
            print(f"- {COVID_LABELS[i]:<20}: {p:.1%}")
    else:
        model = load_nih_model()
        predictions, all_probs = predict_nih(IMAGE_PATH, model)
        if predictions:
            print("✅ Diagnosed conditions (above threshold):")
            for condition, prob in predictions:
                print(f"- {condition:<20}: {prob:.1%}")
        else:
            print("✅ No abnormalities detected above threshold")
        print("\n🔍 Top 3 predictions:")
        top3_indices = np.argsort(all_probs)[-3:][::-1]
        for idx in top3_indices:
            print(f"- {NIH_LABELS[idx]:<20}: {all_probs[idx]:.1%}")

    print("\n✅ Diagnosis complete.")

import torch
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np

MODEL_PATH = "model/densenet121.pth"
IMAGE_PATH = "test.png"
THRESHOLD = 0.5 

DISEASE_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'No Finding'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("🔍 Loading model...")

model = models.densenet121(weights=None) 

model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.3),
    torch.nn.Linear(1024, 256),
    torch.nn.ReLU(inplace=True),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(256, len(DISEASE_LABELS))
)

state_dict = torch.load(MODEL_PATH, map_location=device)

model_state_dict = model.state_dict()

pretrained_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}

model_state_dict.update(pretrained_dict)

model.load_state_dict(model_state_dict)

model = model.to(device)
model.eval()
print("✅ Model loaded successfully!")
print(f"Model architecture:\n{model.classifier}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict(image_path, threshold=0.5):
    """Make prediction on a single image"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]

    predictions = []
    for i, prob in enumerate(probs):
        if prob >= threshold:
            predictions.append((DISEASE_LABELS[i], float(prob)))

    predictions.sort(key=lambda x: x[1], reverse=True)

    return predictions, probs

if __name__ == "__main__":
    print("\n🩺 Running chest X-ray diagnosis...")
    
    predictions, all_probs = predict(IMAGE_PATH, THRESHOLD)
    
    if predictions:
        print("\n✅ Diagnosed conditions:")
        for condition, prob in predictions:
            print(f"- {condition:<20}: {prob:.1%} confidence")
    else:
        print("\n✅ No abnormalities detected above threshold")
    
    top3_indices = np.argsort(all_probs)[-3:][::-1]
    print("\n🔍 Top 3 predictions:")
    for idx in top3_indices:
        print(f"- {DISEASE_LABELS[idx]:<20}: {all_probs[idx]:.1%}")
    
    print("\nDiagnosis complete.")
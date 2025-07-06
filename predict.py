import torch
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# ----------- CONFIG ----------- #
MODEL_PATH = "model/densenet121.pth"
BASE_DATA_DIR = "small-dataset/"  # Root directory containing images_001, images_002, etc.
THRESHOLD = 0.5
REPORT_PATH = "xray_predictions_report.csv"
ERROR_ANALYSIS_DIR = "error_analysis/"  # To save misclassified images

# Must match training labels
DISEASE_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'No Finding'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------- MODEL LOADING ----------- #
def load_model():
    print("🔍 Loading model...")
    model = models.densenet121(weights=None)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(1024, 256),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(256, len(DISEASE_LABELS))
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()
    print("✅ Model loaded successfully!")
    return model

# ----------- IMAGE PROCESSING ----------- #
def find_xray_images(base_dir):
    """Locate all X-ray images in nested NIH folder structure"""
    image_paths = []
    for root in [f"{base_dir}images_{i:03d}/images" for i in range(1, 5)]:
        if os.path.exists(root):
            for img in os.listdir(root):
                if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append((img, os.path.join(root, img)))
    return image_paths

# ----------- PREDICTION & ANALYSIS ----------- #
def analyze_predictions(model, image_paths, true_labels_csv=None):
    """Run batch prediction and generate reports"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load ground truth if available
    df_true = None
    if true_labels_csv and os.path.exists(true_labels_csv):
        df_true = pd.read_csv(true_labels_csv)
        df_true['True_Labels'] = df_true['Finding Labels'].str.split('|')
        os.makedirs(ERROR_ANALYSIS_DIR, exist_ok=True)

    results = []
    for img_name, img_path in tqdm(image_paths, desc="Analyzing X-rays"):
        try:
            image = Image.open(img_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                probs = torch.sigmoid(model(tensor))[0].cpu().numpy()
            
            pred_label = DISEASE_LABELS[np.argmax(probs)]
            confidence = np.max(probs)
            
            # Ground truth comparison
            true_labels = None
            if df_true is not None:
                true_entry = df_true[df_true['Image Index'] == img_name]
                if not true_entry.empty:
                    true_labels = true_entry['True_Labels'].iloc[0]
                    is_correct = pred_label in true_labels
                    
                    # Save misclassified images
                    if not is_correct:
                        error_path = f"{ERROR_ANALYSIS_DIR}{img_name}"
                        image.save(error_path)

            results.append({
                'image_name': img_name,
                'image_path': img_path,
                'predicted_label': pred_label,
                'confidence': confidence,
                'true_labels': true_labels if df_true is not None else "N/A",
                'is_correct': is_correct if df_true is not None else "N/A",
                **dict(zip(DISEASE_LABELS, probs))  # All probabilities
            })
            
        except Exception as e:
            print(f"\n⚠️ Error processing {img_name}: {str(e)}")
    
    return pd.DataFrame(results)

# ----------- REPORT GENERATION ----------- #
def generate_reports(results_df):
    """Generate CSV report and visualizations"""
    # Save full results
    results_df.to_csv(REPORT_PATH, index=False)
    
    # Generate summary stats
    if 'is_correct' in results_df.columns:
        accuracy = results_df['is_correct'].mean()
        print(f"\n📊 Model Accuracy: {accuracy:.2%}")
        
        # Confusion matrix (simplified)
        plt.figure(figsize=(12, 6))
        results_df['predicted_label'].value_counts().plot(kind='bar')
        plt.title("Predicted Conditions Distribution")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("prediction_distribution.png")
        
        # Confidence analysis
        plt.figure(figsize=(10, 5))
        results_df['confidence'].hist(bins=20)
        plt.title("Prediction Confidence Distribution")
        plt.savefig("confidence_distribution.png")

# ----------- MAIN EXECUTION ----------- #
if __name__ == "__main__":
    # 1. Load model
    model = load_model()
    
    # 2. Locate all X-ray images
    print("\n📂 Scanning for X-ray images...")
    xray_images = find_xray_images(BASE_DATA_DIR)
    print(f"Found {len(xray_images)} X-ray images")
    
    # 3. Run predictions
    results_df = analyze_predictions(
        model, 
        xray_images,
        true_labels_csv="Data_Entry_2017.csv"  # Optional
    )
    
    # 4. Generate reports
    generate_reports(results_df)
    
    print(f"\n✅ Analysis complete! Results saved to:")
    print(f"- Full report: {REPORT_PATH}")
    print(f"- Error images: {ERROR_ANALYSIS_DIR}")
    print(f"- Visualization: prediction_distribution.png")
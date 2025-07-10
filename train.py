import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class Config:
    DRIVE_PATH = '/content/small-dataset/small-dataset'
    CSV_PATH = os.path.join('/content/drive/MyDrive/Data_Entry_2017.csv')
    IMAGE_FOLDERS = [
        os.path.join(DRIVE_PATH, 'images_001/images'),
        os.path.join(DRIVE_PATH, 'images_002/images'),
        os.path.join(DRIVE_PATH, 'images_003/images'),
        os.path.join(DRIVE_PATH, 'images_004/images')
    ]
    
    IMG_SIZE = 224
    BATCH_SIZE = 64  
    NUM_EPOCHS = 25  
    LEARNING_RATE = 0.001  
    NUM_CLASSES = 15
    
    TRAIN_RATIO = 0.8 
    VAL_RATIO = 0.1   
    TEST_RATIO = 0.1   
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    MODEL_SAVE_PATH = '/content/densenet121_xray_optimized.pth'
    
    NUM_WORKERS = 4  
    PIN_MEMORY = True
    MIXED_PRECISION = True  

print(f"Using device: {Config.DEVICE}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

DISEASE_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'No Finding'
]

class ChestXrayDataset(Dataset):
    
    def __init__(self, dataframe, image_folders, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        
        print("Building image path cache...")
        self.image_paths = {}
        for folder in tqdm(image_folders, desc="Scanning folders"):
            if os.path.exists(folder):
                for img_name in os.listdir(folder):
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths[img_name] = os.path.join(folder, img_name)
        
        existing_images = set(self.image_paths.keys())
        self.dataframe = self.dataframe[self.dataframe['Image Index'].isin(existing_images)].reset_index(drop=True)
        print(f"Dataset size after filtering: {len(self.dataframe)}")
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_name = row['Image Index']
        labels = row['encoded_labels']
        
        img_path = self.image_paths[img_name]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        labels = torch.FloatTensor(labels)
        
        return image, labels
def prepare_data():
    print("Loading dataset...")

    df = pd.read_csv(Config.CSV_PATH)
    print(f"Total samples in CSV: {len(df)}")

    def parse_labels(finding_labels):
        if pd.isna(finding_labels) or finding_labels == '':
            return ['No Finding']
        else:
            return [label.strip() for label in finding_labels.split('|')]

    df['disease_labels'] = df['Finding Labels'].apply(parse_labels)

    mlb = MultiLabelBinarizer(classes=DISEASE_LABELS)
    encoded_labels = mlb.fit_transform(df['disease_labels'])
    df['encoded_labels'] = list(encoded_labels)

    print("Checking available images...")
    image_paths = {}
    for folder in Config.IMAGE_FOLDERS:
        if os.path.exists(folder):
            for img_name in os.listdir(folder):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    image_paths[img_name] = os.path.join(folder, img_name)

    df = df[df['Image Index'].isin(image_paths.keys())].reset_index(drop=True)
    print(f"Samples with existing images: {len(df)}")

    label_counts = df['Finding Labels'].value_counts()
    valid_labels = label_counts[label_counts >= 2].index
    df = df[df['Finding Labels'].isin(valid_labels)].reset_index(drop=True)
    print(f"Samples after strict filtering: {len(df)}")

    if len(df) > 50000:
        df = df.sample(n=50000, random_state=42).reset_index(drop=True)
        print(f"Sampled {len(df)} images for faster training")

    try:
        train_df, temp_df = train_test_split(
            df,
            test_size=(Config.VAL_RATIO + Config.TEST_RATIO),
            random_state=42,
            stratify=df['Finding Labels']
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=Config.TEST_RATIO / (Config.VAL_RATIO + Config.TEST_RATIO),
            random_state=42,
            stratify=temp_df['Finding Labels']
        )
    except ValueError as e:
        print("❌ Stratified split failed even after filtering. Falling back to random split.")
        train_df, temp_df = train_test_split(
            df,
            test_size=(Config.VAL_RATIO + Config.TEST_RATIO),
            random_state=42,
            stratify=None
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=Config.TEST_RATIO / (Config.VAL_RATIO + Config.TEST_RATIO),
            random_state=42,
            stratify=None
        )

    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

    return train_df, val_df, test_df



def get_transforms():
    
    train_transforms = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_transforms

def create_model():
    model = models.densenet121(pretrained=True)
    
    for param in list(model.parameters())[:-10]:  
        param.requires_grad = False
    
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 256), 
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, Config.NUM_CLASSES)
    )
    
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
   
    scaler = torch.cuda.amp.GradScaler() if Config.MIXED_PRECISION else None
    
    best_val_loss = float('inf')
    best_model_state = None
    patience = 5 
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_f1_scores = []
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 50)
        
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for images, labels in tqdm(train_loader, desc='Training'):
            images = images.to(Config.DEVICE, non_blocking=True)
            labels = labels.to(Config.DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            if Config.MIXED_PRECISION and scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validation'):
                images = images.to(Config.DEVICE, non_blocking=True)
                labels = labels.to(Config.DEVICE, non_blocking=True)
                
                if Config.MIXED_PRECISION:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_batches += 1
                
                predictions = torch.sigmoid(outputs) > 0.5
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        val_accuracy = accuracy_score(all_labels, all_predictions)
        val_f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
        
        val_accuracies.append(val_accuracy)
        val_f1_scores.append(val_f1)
        
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}')
        print(f'Val Accuracy: {val_accuracy:.4f}')
        print(f'Val F1 Score: {val_f1:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f'✓ New best validation loss: {best_val_loss:.4f}')
        else:
            patience_counter += 1
            print(f'No improvement for {patience_counter} epochs')
            
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses, val_accuracies, val_f1_scores

def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    test_batches = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images = images.to(Config.DEVICE, non_blocking=True)
            labels = labels.to(Config.DEVICE, non_blocking=True)
            
            if Config.MIXED_PRECISION:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            test_batches += 1
            
            predictions = torch.sigmoid(outputs) > 0.5
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_test_loss = test_loss / test_batches
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    test_accuracy = accuracy_score(all_labels, all_predictions)
    test_f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    
    return avg_test_loss, test_accuracy, test_f1

def plot_training_history(train_losses, val_losses, val_accuracies, val_f1_scores):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(train_losses, label='Train Loss', color='blue')
    axes[0, 0].plot(val_losses, label='Validation Loss', color='red')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(val_accuracies, label='Validation Accuracy', color='green')
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(val_f1_scores, label='Validation F1 Score', color='orange')
    axes[1, 0].set_title('Validation F1 Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(train_losses, label='Train Loss', color='blue', alpha=0.7)
    axes[1, 1].plot(val_losses, label='Val Loss', color='red', alpha=0.7)
    axes[1, 1].set_title('Learning Curves')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('/content/training_history_optimized.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("=" * 60)
    print("OPTIMIZED CHEST X-RAY DETECTION TRAINING")
    print("Using DenseNet121 with first 4 image folders")
    print("=" * 60)
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    train_df, val_df, test_df = prepare_data()
    
    train_transforms, val_transforms = get_transforms()
    
    train_dataset = ChestXrayDataset(train_df, Config.IMAGE_FOLDERS, train_transforms)
    val_dataset = ChestXrayDataset(val_df, Config.IMAGE_FOLDERS, val_transforms)
    test_dataset = ChestXrayDataset(test_df, Config.IMAGE_FOLDERS, val_transforms)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        persistent_workers=True
    )
    
    print(f"✓ Data loaders created successfully!")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    model = create_model()
    model = model.to(Config.DEVICE)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5, verbose=True
    )
    
    print(f"✓ Model created: DenseNet121")
    print(f"✓ Device: {Config.DEVICE}")
    print(f"✓ Mixed Precision: {Config.MIXED_PRECISION}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    print("\n🚀 Starting optimized training...")
    trained_model, train_losses, val_losses, val_accuracies, val_f1_scores = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, Config.NUM_EPOCHS
    )
    
    torch.save(trained_model.state_dict(), Config.MODEL_SAVE_PATH)
    print(f"\n✓ Model saved to {Config.MODEL_SAVE_PATH}")
    
    print("\n📊 Evaluating on test set...")
    test_loss, test_accuracy, test_f1 = evaluate_model(trained_model, test_loader, criterion)
    
    print(f"\n🎯 FINAL TEST RESULTS:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    
    plot_training_history(train_losses, val_losses, val_accuracies, val_f1_scores)
    
    print("\n" + "=" * 60)
    print("✅ OPTIMIZED TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main() 
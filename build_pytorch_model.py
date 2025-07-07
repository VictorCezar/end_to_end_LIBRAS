import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import ffmpeg
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Importa o modelo I3D que você já usava
from pytorchvideo.models.hub import i3d_r50

# --- Configurações Globais (Do seu código original) ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_FRAMES = 32
BATCH_SIZE = 8
NUM_CLASSES = 20
RANDOM_SEED = 42
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100 
PATIENCE = 10 # Paciência original

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 1. Funções Auxiliares para Decodificação de Vídeo (Seu código original) ---
def decode_video_from_path_ffmpeg(video_path_str, num_frames=NUM_FRAMES, img_height=IMG_HEIGHT, img_width=IMG_WIDTH):
    video_path = video_path_str
    try:
        out, err = (
            ffmpeg
            .input(video_path)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{img_width}x{img_height}')
            .run(capture_stdout=True, capture_stderr=True, quiet=True)
        )
        frames_decoded = np.frombuffer(out, np.uint8).reshape([-1, img_height, img_width, 3])
        total_decoded_frames = frames_decoded.shape[0]

        if total_decoded_frames == 0:
            return np.zeros((num_frames, img_height, img_width, 3), dtype=np.float32), 0
        
        # Amostragem de frames
        if total_decoded_frames < num_frames:
            sampled_indices = np.arange(total_decoded_frames)
            padding_needed = num_frames - total_decoded_frames
        else:
            sampled_indices = np.linspace(0, total_decoded_frames - 1, num_frames, dtype=int)
            padding_needed = 0

        video_array = frames_decoded[sampled_indices].astype(np.float32) / 255.0

        if padding_needed > 0:
            padding = np.zeros((padding_needed, img_height, img_width, 3), dtype=np.float32)
            video_array = np.concatenate([video_array, padding], axis=0)
        
        return video_array, 1
    except (ffmpeg.Error, Exception) as e:
        return np.zeros((num_frames, img_height, img_width, 3), dtype=np.float32), 0

# --- 2. Custom PyTorch Dataset (Seu código original, sem Augmentation complexa) ---
class VideoDataset(Dataset):
    def __init__(self, metadata_df, is_training=True):
        self.is_training = is_training
        self.video_data_cache = []
        self.labels_cache = []

        print(f"Initializing {('training' if is_training else 'validation/test')} dataset. Decoding and storing videos in RAM...")
        for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
            video_path = row['video_path']
            label_id = row['label_id']
            video_array, is_valid_flag = decode_video_from_path_ffmpeg(video_path, NUM_FRAMES, IMG_HEIGHT, IMG_WIDTH)
            if is_valid_flag == 1:
                self.video_data_cache.append(video_array)
                self.labels_cache.append(label_id)
        
        print(f"Total valid videos in {('training' if is_training else 'validation/test')} dataset: {len(self.video_data_cache)}")

    def __len__(self):
        return len(self.video_data_cache)

    def __getitem__(self, idx):
        video_array = self.video_data_cache[idx]
        label = self.labels_cache[idx]
        video_tensor = torch.from_numpy(video_array).float()

        if self.is_training and torch.rand(1).item() < 0.5:
            video_tensor = torch.flip(video_tensor, [2])
            
        return video_tensor, torch.tensor(label, dtype=torch.long)

# --- 3. Definição do Modelo (Seu código original) ---
def build_i3d_model(num_classes=NUM_CLASSES):
    model = i3d_r50(pretrained=True) 
    num_ftrs = model.blocks[-1].proj.in_features
    model.blocks[-1].proj = nn.Linear(num_ftrs, num_classes)
    return model

# --- 4. Loop de Treinamento e Avaliação (Seu código original, monitorando Val Loss) ---
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=NUM_EPOCHS, model_save_path="best_model.pth", patience=PATIENCE):
    print("\nStarting training...")
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for videos, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)"):
            videos, labels = videos.to(DEVICE), labels.to(DEVICE)
            videos = videos.permute(0, 4, 1, 2, 3)
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * videos.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100 * correct_train / total_train
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")

        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, desc=" (Validation)") 
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"*** Saved best model at epoch {epoch+1} with Val Loss: {best_val_loss:.4f} (Acc: {val_acc:.2f}%) ***")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break 
    
    return history

def evaluate_model(model, data_loader, criterion, desc=" (Evaluation)"):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_labels, all_predictions = [], []
    
    with torch.no_grad():
        for videos, labels in tqdm(data_loader, desc=f"Evaluation{desc}"):
            videos, labels = videos.to(DEVICE), labels.to(DEVICE)
            videos = videos.permute(0, 4, 1, 2, 3)
            outputs = model(videos)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * videos.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = 100 * correct / total
    print(f"Evaluation{desc} Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")
    return avg_loss, accuracy, all_labels, all_predictions

# --- Script Principal ---
if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    base_splits_dir = "/data/popo/dataset_splits"
    train_csv = os.path.join(base_splits_dir, "train_metadata.csv")
    val_csv = os.path.join(base_splits_dir, "val_metadata.csv")
    test_csv = os.path.join(base_splits_dir, "test_metadata.csv")

    train_df = pd.read_csv(train_csv); val_df = pd.read_csv(val_csv); test_df = pd.read_csv(test_csv)
    
    train_dataset = VideoDataset(train_df, is_training=True)
    val_dataset = VideoDataset(val_df, is_training=False)
    test_dataset = VideoDataset(test_df, is_training=False)

    # ### CORREÇÃO APLICADA AQUI ###
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=False)
    
    model = build_i3d_model(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model_save_dir = os.path.join(base_splits_dir, "saved_models_english")
    os.makedirs(model_save_dir, exist_ok=True)
    best_model_path = os.path.join(model_save_dir, "best_i3d_model_english.pth")
    
    history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=NUM_EPOCHS, model_save_path=best_model_path, patience=PATIENCE)

    history_save_path = os.path.join(model_save_dir, 'training_history_english.npy')
    np.save(history_save_path, history)
    print(f"Training history saved to: {history_save_path}")

    print("\nEvaluating the best model on the final test set...")
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc, true_labels, predicted_labels = evaluate_model(model, test_loader, criterion, desc=" (Test)")
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_acc:.2f}%")

    # ### SEÇÃO DE PLOTS EM INGLÊS ###
    
    plots_dir = os.path.join(base_splits_dir, "paper_plots_english")
    os.makedirs(plots_dir, exist_ok=True)
    
    label_mapping_path = os.path.join(base_splits_dir, "label_mapping.txt")
    id_to_label = {int(line.strip().split(':')[1]): line.strip().split(':')[0] for line in open(label_mapping_path)}
    class_names = [id_to_label[i] for i in sorted(id_to_label.keys())]

    print("\nGenerating Classification Report (Precision, Recall, F1-Score)...")
    report = classification_report(true_labels, predicted_labels, target_names=class_names, digits=4)
    print(report)
    report_path = os.path.join(plots_dir, "classification_report_english.txt")
    with open(report_path, "w") as f: f.write(f"Final Accuracy: {test_acc:.4f}%\nFinal Loss: {test_loss:.4f}\n\n" + report)
    print(f"Classification report saved to: {report_path}")

    print("\nGenerating plots for the paper...")
    plt.style.use('seaborn-v0_8-whitegrid')
    
    epochs_ran = len(history['train_loss'])
    if epochs_ran > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        epochs_range = range(1, epochs_ran + 1)
        
        ax1.plot(epochs_range, history['train_loss'], 'o-', label='Training Loss')
        ax1.plot(epochs_range, history['val_loss'], 'o-', label='Validation Loss')
        ax1.set_title('Model Loss History', fontsize=16)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss (Cross-Entropy)', fontsize=12)
        ax1.legend(fontsize=12)
        ax1.grid(True)
        
        ax2.plot(epochs_range, history['train_acc'], 'o-', label='Training Accuracy')
        ax2.plot(epochs_range, history['val_acc'], 'o-', label='Validation Accuracy')
        ax2.set_title('Model Accuracy History', fontsize=16)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.legend(fontsize=12)
        ax2.grid(True)
        
        fig.tight_layout()
        plot_path = os.path.join(plots_dir, "training_history_english.png")
        fig.savefig(plot_path, dpi=300)
        print(f"Training history plot saved to: {plot_path}")
        plt.close(fig)

    cm_normalized = confusion_matrix(true_labels, predicted_labels, labels=sorted(id_to_label.keys()), normalize='true')
    fig_cm, ax_cm = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax_cm)
    
    ax_cm.set_xlabel('Predicted Label', fontsize=14)
    ax_cm.set_ylabel('True Label', fontsize=14)
    ax_cm.set_title('Normalized Confusion Matrix (Test Set)', fontsize=18)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    fig_cm.tight_layout()
    cm_plot_path = os.path.join(plots_dir, "confusion_matrix_normalized_english.png")
    fig_cm.savefig(cm_plot_path, dpi=300)
    print(f"Normalized Confusion Matrix saved to: {cm_plot_path}")
    plt.close(fig_cm)

    print("\nProcess finished successfully!")
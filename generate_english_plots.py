import torch
import torch.nn as nn
# A CORREÇÃO ESTÁ AQUI:
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import ffmpeg
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Importa o modelo I3D
from pytorchvideo.models.hub import i3d_r50

# --- Configurações ---
# Garanta que estas configurações são as mesmas do seu treinamento
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_FRAMES = 32
BATCH_SIZE = 8
NUM_CLASSES = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Funções e Classes do Script Original (Necessárias para carregar os dados e o modelo) ---

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
        if total_decoded_frames == 0: return np.zeros((num_frames, img_height, img_width, 3), dtype=np.float32), 0
        indices = np.linspace(0, total_decoded_frames - 1, num_frames, dtype=int)
        video_array = frames_decoded[indices].astype(np.float32) / 255.0
        if video_array.shape[0] < num_frames:
            padding = np.zeros((num_frames - video_array.shape[0], img_height, img_width, 3), dtype=np.float32)
            video_array = np.concatenate([video_array, padding], axis=0)
        return video_array, 1
    except (ffmpeg.Error, Exception):
        return np.zeros((num_frames, img_height, img_width, 3), dtype=np.float32), 0

class VideoDataset(Dataset):
    def __init__(self, metadata_df):
        self.video_data_cache = []
        self.labels_cache = []
        print(f"Loading test dataset. Decoding and storing videos in RAM...")
        for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
            video_path, label_id = row['video_path'], row['label_id']
            video_array, is_valid_flag = decode_video_from_path_ffmpeg(video_path, NUM_FRAMES, IMG_HEIGHT, IMG_WIDTH)
            if is_valid_flag == 1:
                self.video_data_cache.append(video_array)
                self.labels_cache.append(label_id)
        print(f"Total valid videos in test dataset: {len(self.video_data_cache)}")

    def __len__(self):
        return len(self.video_data_cache)

    def __getitem__(self, idx):
        video_array = self.video_data_cache[idx]
        label = self.labels_cache[idx]
        return torch.from_numpy(video_array).float(), torch.tensor(label, dtype=torch.long)

def build_i3d_model(num_classes=NUM_CLASSES):
    model = i3d_r50(pretrained=True) 
    num_ftrs = model.blocks[-1].proj.in_features
    model.blocks[-1].proj = nn.Linear(num_ftrs, num_classes)
    return model

def evaluate_for_plots(model, data_loader):
    model.eval()
    all_labels, all_predictions = [], []
    with torch.no_grad():
        for videos, labels in tqdm(data_loader, desc="Evaluating on Test Set"):
            videos, labels = videos.to(DEVICE), labels.to(DEVICE)
            videos = videos.permute(0, 4, 1, 2, 3) # (N, C, D, H, W)
            outputs = model(videos)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    return all_labels, all_predictions

# --- Script Principal ---
if __name__ == "__main__":
    base_splits_dir = "/data/popo/dataset_splits"
    
    # --- CAMINHO PARA O SEU MODELO SALVO ---
    # Verifique se o nome do diretório e do arquivo estão corretos
    model_save_dir = os.path.join(base_splits_dir, "saved_models") 
    best_model_path = os.path.join(model_save_dir, "best_i3d_model.pth")
    
    if not os.path.exists(best_model_path):
        print(f"ERROR: Model not found at '{best_model_path}'")
        print("Please check if the path and filename are correct.")
        exit()

    print("Loading test data...")
    test_csv = os.path.join(base_splits_dir, "test_metadata.csv")
    test_df = pd.read_csv(test_csv)
    test_dataset = VideoDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"\nLoading saved model from {best_model_path}...")
    model = build_i3d_model(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(best_model_path))

    true_labels, predicted_labels = evaluate_for_plots(model, test_loader)

    # --- GERAÇÃO DE PLOTS E MÉTRICAS EM INGLÊS ---
    
    plots_dir = os.path.join(base_splits_dir, "paper_plots_english") # Novo diretório
    os.makedirs(plots_dir, exist_ok=True)
    
    label_mapping_path = os.path.join(base_splits_dir, "label_mapping.txt")
    id_to_label = {int(line.strip().split(':')[1]): line.strip().split(':')[0] for line in open(label_mapping_path)}
    class_names = [id_to_label[i] for i in sorted(id_to_label.keys())]

    print("\nGenerating Classification Report (Precision, Recall, F1-Score)...")
    report = classification_report(true_labels, predicted_labels, target_names=class_names, digits=4)
    print(report)
    report_path = os.path.join(plots_dir, "classification_report.txt")
    accuracy_score = 100 * (np.array(true_labels) == np.array(predicted_labels)).sum() / len(true_labels)
    with open(report_path, "w") as f: f.write(f"Final Accuracy: {accuracy_score:.4f}%\n\n" + report)
    print(f"Classification report saved to: {report_path}")

    print("\nGenerating plots for the paper...")
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Matriz de Confusão Normalizada em INGLÊS
    cm_normalized = confusion_matrix(true_labels, predicted_labels, labels=sorted(id_to_label.keys()), normalize='true')
    fig_cm, ax_cm = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax_cm)
    
    # Labels em Inglês
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
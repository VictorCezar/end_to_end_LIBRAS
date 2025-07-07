import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import time
import ffmpeg
from tqdm import tqdm

# Importa o modelo I3D diretamente da biblioteca pytorchvideo
from pytorchvideo.models.hub import i3d_r50

# --- FUNÇÕES NECESSÁRIAS (COPIADAS DO SEU SCRIPT ORIGINAL) ---

def decode_video_from_path_ffmpeg(video_path_str, num_frames=32, img_height=224, img_width=224):
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
            return None, 0
        
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
        return None, 0

def build_i3d_model(num_classes=20):
    model = i3d_r50(pretrained=True) 
    num_ftrs = model.blocks[-1].proj.in_features
    model.blocks[-1].proj = nn.Linear(num_ftrs, num_classes)
    return model

# --- SCRIPT PRINCIPAL DE MEDIÇÃO (CONFIGURADO PARA CPU) ---

if __name__ == "__main__":
    # --- Configurações ---
    NUM_CLASSES = 20
    TARGET_DEVICE = 'cpu' # Força o uso da CPU
    
    # Caminhos
    base_splits_dir = "/data/popo/dataset_splits"
    test_csv = os.path.join(base_splits_dir, "test_metadata.csv")
    model_save_dir = os.path.join(base_splits_dir, "saved_models_english")
    best_model_path = os.path.join(model_save_dir, "best_i3d_model_english.pth")

    # Carregar o modelo e movê-lo para a CPU
    print(f"Loading the trained model onto {TARGET_DEVICE.upper()}...")
    device = torch.device(TARGET_DEVICE)
    model = build_i3d_model(num_classes=NUM_CLASSES)
    
    # map_location=device garante que o modelo seja carregado na CPU
    model.load_state_dict(torch.load(best_model_path, map_location=device)) 
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # Carregar o dataframe do conjunto de teste
    test_df = pd.read_csv(test_csv)
    video_paths = test_df['video_path'].tolist()

    # --- Loop de Medição ---
    data_prep_times = []
    inference_times = []
    
    print(f"Measuring pipeline performance on CPU across {len(video_paths)} test videos...")
    with torch.no_grad():
        for video_path in tqdm(video_paths):
            # --- Parte 1: Medir a preparação do dado ---
            start_prep_time = time.perf_counter()
            video_array, is_valid = decode_video_from_path_ffmpeg(video_path)
            end_prep_time = time.perf_counter()
            
            if not is_valid:
                print(f"Skipping invalid video: {video_path}")
                continue
            
            data_prep_times.append(end_prep_time - start_prep_time)

            # Preparar o tensor para o modelo
            video_tensor = torch.from_numpy(video_array).float()
            video_tensor = video_tensor.permute(3, 0, 1, 2).unsqueeze(0).to(device)

            # --- Parte 2: Medir a inferência do modelo ---
            start_inference_time = time.perf_counter()
            _ = model(video_tensor)
            end_inference_time = time.perf_counter()
            inference_times.append(end_inference_time - start_inference_time)

    # --- Calcular e Exibir os Resultados ---
    avg_prep_ms = np.mean(data_prep_times) * 1000
    std_prep_ms = np.std(data_prep_times) * 1000
    
    avg_inference_ms = np.mean(inference_times) * 1000
    std_inference_ms = np.std(inference_times) * 1000
    
    total_avg_ms = avg_prep_ms + avg_inference_ms
    
    # O hardware da CPU é um Intel Core i7-11700, como especificado no seu artigo 
    print("\n--- CPU Pipeline Performance Results (Intel Core i7-11700) ---")
    print(f"Part 1 - Data Preparation (FFMPEG Decode, Resize, Sample): {avg_prep_ms:.2f} \u00B1 {std_prep_ms:.2f} ms")
    print(f"Part 2 - Model Inference (CPU):                       {avg_inference_ms:.2f} \u00B1 {std_inference_ms:.2f} ms")
    print("------------------------------------------------------------------")
    print(f"Total End-to-End Time per Video:                           {total_avg_ms:.2f} ms")
    total_fps = 1000 / total_avg_ms
    print(f"Total Pipeline Throughput:                                 ~{total_fps:.2f} FPS")
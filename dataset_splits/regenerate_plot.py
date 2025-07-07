import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import ConnectionPatch

# --- Configuration: Set the paths to your saved files ---
base_splits_dir = "/data/popo/dataset_splits"
model_save_dir = os.path.join(base_splits_dir, "saved_models_english")
plots_dir = os.path.join(base_splits_dir, "paper_plots_english")
os.makedirs(plots_dir, exist_ok=True)

history_path = os.path.join(model_save_dir, 'training_history_english.npy')
# Salvar com um novo nome para a versão final
output_plot_path = os.path.join(plots_dir, "training_history_final_v2.png") 

# --- Configuration for the Zoom ---
ZOOM_START_EPOCH = 15 

print(f"Loading training history from: {history_path}")

# --- 1. Load the Saved History Data ---
try:
    history = np.load(history_path, allow_pickle=True).item()
    print("History loaded successfully.")
    epochs_ran = len(history['train_loss'])
    print(f"Data contains {epochs_ran} epochs.")
except FileNotFoundError:
    print(f"ERROR: History file not found at {history_path}")
    exit()

# --- 2. Generate the Plot with Final Adjustments ---
if epochs_ran > 0:
    print("Generating new plot with final layout...")
    plt.style.use('seaborn-v0_8-whitegrid')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8))
    epochs_range = range(1, epochs_ran + 1)
    
    # --- Main Plot 1: Loss History ---
    ax1.plot(epochs_range, history['train_loss'], 'o-', label='Training Loss', color='royalblue', linewidth=2.5)
    ax1.plot(epochs_range, history['val_loss'], 'o-', label='Validation Loss', color='red', linewidth=2.5)
    ax1.set_title('Model Loss History', fontsize=28)
    ax1.set_xlabel('Epoch', fontsize=22)
    ax1.set_ylabel('Loss (Cross-Entropy)', fontsize=22)
    ax1.legend(fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=18)
    ax1.grid(True)

    # --- Main Plot 2: Accuracy History ---
    ax2.plot(epochs_range, history['train_acc'], 'o-', label='Training Accuracy', color='royalblue', linewidth=2.5)
    ax2.plot(epochs_range, history['val_acc'], 'o-', label='Validation Accuracy', color='red', linewidth=2.5)
    ax2.set_title('Model Accuracy History', fontsize=28)
    ax2.set_xlabel('Epoch', fontsize=22)
    ax2.set_ylabel('Accuracy (%)', fontsize=22)
    ax2.legend(fontsize=20, loc='lower right') # Mantendo a legenda aqui por enquanto
    ax2.tick_params(axis='both', which='major', labelsize=18)
    ax2.grid(True)

    # --- 3. Create the Inset Zooms (if possible) ---
    if epochs_ran > ZOOM_START_EPOCH:
        zoom_epoch_range = range(ZOOM_START_EPOCH, epochs_ran + 1)
        zoom_epoch_indices = range(ZOOM_START_EPOCH - 1, epochs_ran)

        # Inset for Loss Plot - EXPANDIDO
        axins1 = ax1.inset_axes([0.3, 0.35, 0.6, 0.5]) 
        axins1.plot(zoom_epoch_range, np.array(history['train_loss'])[zoom_epoch_indices], 'o-', color='royalblue', linewidth=2.5)
        axins1.plot(zoom_epoch_range, np.array(history['val_loss'])[zoom_epoch_indices], 'o-', color='red', linewidth=2.5)
        
        y1_min = min(np.array(history['train_loss'])[zoom_epoch_indices])
        y1_max = max(np.array(history['val_loss'])[zoom_epoch_indices])
        axins1.set_ylim(y1_min * 0.98, y1_max * 1.02)
        axins1.grid(True, linestyle='--')
        axins1.tick_params(axis='both', which='major', labelsize=16)
        ax1.indicate_inset_zoom(axins1, edgecolor="black")

        # Inset for Accuracy Plot - EXPANDIDO E MOVIDO PARA CIMA
        # MUDANÇA AQUI: A coordenada y foi de 0.15 para 0.22 para subir o gráfico
        axins2 = ax2.inset_axes([0.3, 0.22, 0.6, 0.5])
        axins2.plot(zoom_epoch_range, np.array(history['train_acc'])[zoom_epoch_indices], 'o-', color='royalblue', linewidth=2.5)
        axins2.plot(zoom_epoch_range, np.array(history['val_acc'])[zoom_epoch_indices], 'o-', color='red', linewidth=2.5)
        
        y2_min = min(np.array(history['val_acc'])[zoom_epoch_indices])
        y2_max = max(np.array(history['train_acc'])[zoom_epoch_indices])
        axins2.set_ylim(y2_min * 0.99, y2_max * 1.01)
        axins2.grid(True, linestyle='--')
        axins2.tick_params(axis='both', which='major', labelsize=16)
        ax2.indicate_inset_zoom(axins2, edgecolor="black")

    # Save the final figure
    fig.tight_layout()
    fig.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    print(f"New plot with final layout v2 saved successfully to: {output_plot_path}")
    plt.close(fig)
else:
    print("The loaded history is empty. Cannot generate plot.")
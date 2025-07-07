import tensorflow as tf
import pandas as pd
import numpy as np
import os
# import cv2 # Remover/comentar
# import tensorflow_io as tfio # Remover/comentar
import ffmpeg # NOVO: Importar ffmpeg-python

# --- Configurações Globais ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_FRAMES = 32
BATCH_SIZE = 8
NUM_CLASSES = 20
RANDOM_SEED = 42

tf.keras.mixed_precision.set_global_policy('mixed_float16')
print(f"Política de precisão global definida como: {tf.keras.mixed_precision.global_policy().name}")


# --- 1. Funções Auxiliares para Pré-processamento ---

# NOVO: Função para decodificar vídeo usando ffmpeg-python
def decode_video_from_path_ffmpeg(video_path_str, num_frames=NUM_FRAMES, img_height=IMG_HEIGHT, img_width=IMG_WIDTH):
    """
    Decodifica um vídeo usando ffmpeg-python, amostrando um número fixo de quadros.

    Args:
        video_path_str (bytes): Caminho do arquivo de vídeo (como bytes de tf.py_function).
        num_frames (int): Número de quadros a amostrar.
        img_height (int): Altura desejada dos quadros.
        img_width (int): Largura desejada dos quadros.

    Returns:
        tuple: (tf.Tensor, int) - Um tensor de quadros de vídeo pré-processados e um flag (1 para sucesso, 0 para falha).
                                Retorna um tensor de zeros e flag 0 se o vídeo não puder ser lido.
    """
    video_path = video_path_str.numpy().decode('utf-8')
    
    try:
        # Pega a duração do vídeo para determinar os quadros a amostrar
        probe = ffmpeg.probe(video_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        # Use duration_ts e r_frame_rate para calcular total_frames com mais precisão
        # Ou tente 'nb_frames' diretamente, mas pode ser 0 para alguns formatos
        
        # Calcular total_frames a partir da duração e taxa de quadros
        duration_seconds = float(video_info['duration'])
        avg_frame_rate_str = video_info['avg_frame_rate'] # Ex: "30000/1001"
        num, den = map(int, avg_frame_rate_str.split('/'))
        avg_fps = num / den
        
        total_frames = int(duration_seconds * avg_fps)
        
        if total_frames == 0:
            # Fallback se a estimativa for 0, pode ser que nb_frames funcione
            if 'nb_frames' in video_info:
                total_frames = int(video_info['nb_frames'])
            if total_frames == 0:
                print(f"Aviso: Vídeo vazio ou sem quadros decodificados com FFmpeg: {video_path}. Retornando zeros.")
                return tf.zeros((num_frames, img_height, img_width, 3), dtype=tf.float32), 0

        # Determine os índices dos quadros a serem amostrados uniformemente
        if total_frames < num_frames:
            # Se menos quadros que o desejado, amostre todos disponíveis e preencha com zeros
            indices_to_sample = np.arange(total_frames)
            padding_needed = num_frames - total_frames
        else:
            indices_to_sample = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            padding_needed = 0

        # Prepara o filtro `select` do FFmpeg para extrair apenas os quadros desejados
        # e redimensiona-os diretamente no FFmpeg para eficiência
        select_filter = ','.join([f'eq(n,{idx})' for idx in indices_to_sample])

        out, err = (
            ffmpeg
            .input(video_path)
            .filter('select', select_filter) # Seleciona os quadros desejados
            .filter('scale', img_width, img_height) # Redimensiona
            .output('pipe:', format='rawvideo', pix_fmt='rgb24') # Saída em formato raw, RGB
            .run(capture_stdout=True, capture_stderr=True)
        )
        
        # Converte a saída raw para um array NumPy e depois para um tensor TensorFlow
        frames = np.frombuffer(out, np.uint8).reshape([-1, img_height, img_width, 3])
        video_tensor = tf.convert_to_tensor(frames, dtype=tf.float32) / 255.0 # Normaliza para [0,1]

        # Aplicar padding se o video era muito curto
        if padding_needed > 0:
             padding = tf.zeros((padding_needed, img_height, img_width, 3), dtype=tf.float32)
             video_tensor = tf.concat([video_tensor, padding], axis=0)

        # Certificar que o tensor tem o número exato de quadros (pode variar ligeiramente devido a linspace/casting)
        if tf.shape(video_tensor)[0] > num_frames:
            video_tensor = video_tensor[:num_frames]
        elif tf.shape(video_tensor)[0] < num_frames:
             padding = tf.zeros((num_frames - tf.shape(video_tensor)[0], img_height, img_width, 3), dtype=tf.float32)
             video_tensor = tf.concat([video_tensor, padding], axis=0)
        
        return video_tensor, 1 # Retorna o tensor e flag de sucesso

    except ffmpeg.Error as e:
        # Erro específico do FFmpeg, imprimir stderr para diagnóstico
        print(f"Erro FFmpeg ao processar vídeo {video_path}: {e.stderr.decode('utf8')}. Retornando zeros.")
        return tf.zeros((num_frames, img_height, img_width, 3), dtype=tf.float32), 0
    except Exception as e:
        # Outros erros inesperados
        print(f"Erro inesperado ao processar vídeo {video_path}: {e}. Retornando zeros.")
        return tf.zeros((num_frames, img_height, img_width, 3), dtype=tf.float32), 0


def _parse_function(video_path_tensor, label_id_tensor, is_training=True):
    """
    Função de parsing para o tf.data.Dataset.
    Decodifica o vídeo e aplica pré-processamento/aumento de dados.
    """
    # CHAME A NOVA FUNÇÃO DE DECODIFICAÇÃO AQUI
    video_tensor, is_valid_flag = tf.py_function(
        func=decode_video_from_path_ffmpeg, # Mudar para a nova função
        inp=[video_path_tensor, NUM_FRAMES, IMG_HEIGHT, IMG_WIDTH],
        Tout=[tf.float32, tf.int32] # Espera um tensor float32 e um int32
    )
    video_tensor.set_shape([NUM_FRAMES, IMG_HEIGHT, IMG_WIDTH, 3])
    is_valid_flag.set_shape([]) # Flag é um escalar

    # Aplicar Data Augmentation (apenas no conjunto de treino)
    if is_training:
        # ATENÇÃO: random_flip_left_right pode ser problemático para alguns sinais. Avalie.
        video_tensor = tf.image.random_flip_left_right(video_tensor, seed=RANDOM_SEED)
        video_tensor = tf.image.random_brightness(video_tensor, max_delta=0.1, seed=RANDOM_SEED)
        video_tensor = tf.image.random_contrast(video_tensor, lower=0.9, upper=1.1, seed=RANDOM_SEED)
        
    return video_tensor, label_id_tensor, is_valid_flag # Retorna a flag

# ... (o restante do script: create_video_dataset, build_3d_cnn_model e o if __name__ == "__main__":, permanece o mesmo)
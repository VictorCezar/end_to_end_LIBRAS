import os
import pandas as pd
import re

def extract_video_metadata(base_dir="/data/popo"):
    """
    Extrai metadados dos vídeos do dataset MINDS-Libras e os organiza em um DataFrame.

    Args:
        base_dir (str): O diretório base onde as pastas SinalizadorXX estão localizadas.

    Returns:
        pd.DataFrame: Um DataFrame contendo metadados dos vídeos, como caminho,
                      nome do sinal, ID do sinalizador e ID da repetição.
    """
    data = []
    
    # Mapeamento do número do sinal para o nome do sinal
    signal_id_to_name = {
        '01': 'Acontecer', '02': 'Aluno', '03': 'Amarelo', '04': 'America',
        '05': 'Aproveitar', '06': 'Bala', '07': 'Banco', '08': 'Banheiro',
        '09': 'Barulho', '10': 'Cinco', '11': 'Conhecer', '12': 'Espelho',
        '13': 'Esquina', '14': 'Filho', '15': 'Maca', '16': 'Medo',
        '17': 'Ruim', '18': 'Sapo', '19': 'Vacina', '20': 'Vontade'
    }

    # Padrão de expressão regular para extrair informações do nome do arquivo
    # Grupo 1: ID do Sinal (XX)
    # Grupo 2: Nome do Sinal (SinalNome)
    # Grupo 3: ID do Sinalizador (YY)
    # Grupo 4: Repetição (Z)
    # Grupo 5: Sufixo opcional (_X)
    video_name_pattern = re.compile(r'^(\d{2})([A-Za-z]+)Sinalizador(\d{2})-(\d)(_\d+)?\.mp4$')

    print(f"Varrendo diretórios a partir de: {base_dir}")

    for i in range(1, 13): # Sinalizador01 a Sinalizador12
        sinalizador_id_dir = f"{i:02d}" # ID do sinalizador baseado no nome do diretório
        sinalizador_path = os.path.join(base_dir, f"Sinalizador{sinalizador_id_dir}", "Canon")

        if not os.path.isdir(sinalizador_path):
            print(f"Aviso: Diretório não encontrado para Sinalizador{sinalizador_id_dir} em {sinalizador_path}")
            continue

        print(f"Processando diretório: {sinalizador_path}")

        for filename in os.listdir(sinalizador_path):
            if filename.endswith(".mp4"):
                match = video_name_pattern.match(filename)
                if match:
                    sinal_id_from_name = match.group(1)
                    sinal_name_from_file = match.group(2)
                    sinalizador_id_from_file = match.group(3)
                    repeticao_id = int(match.group(4))
                    suffix = match.group(5) # O sufixo (_1, _2, etc.) ou None

                    full_video_path = os.path.join(sinalizador_path, filename)

                    # Verificar consistência do ID do sinalizador e nome do sinal
                    if sinalizador_id_from_file == sinalizador_id_dir and \
                       signal_id_to_name.get(sinal_id_from_name) == sinal_name_from_file:
                        
                        data.append({
                            'video_path': full_video_path,
                            'sinal_id': sinal_id_from_name,
                            'sinal_nome': sinal_name_from_file,
                            'sinalizador_id': sinalizador_id_from_file,
                            'repeticao_id': repeticao_id,
                            'has_suffix': bool(suffix) # True se tiver sufixo (_1, _2 etc), False caso contrário
                        })
                    else:
                        print(f"Aviso: Inconsistência no nome do arquivo: {filename}. Ignorando.")
                else:
                    print(f"Aviso: Nome de arquivo inesperado que não corresponde ao padrão: {filename}. Ignorando.")
    
    df = pd.DataFrame(data)
    
    # --- Lógica para lidar com duplicatas e priorizar arquivos sem sufixo ---
    # Ordena para garantir que arquivos sem sufixo (`has_suffix=False`) venham primeiro
    # em caso de duplicatas em (sinal_id, sinalizador_id, repeticao_id)
    df_sorted = df.sort_values(by=['sinal_id', 'sinalizador_id', 'repeticao_id', 'has_suffix'])
    
    # Remove duplicatas, mantendo a primeira ocorrência (que será a sem sufixo, devido à ordenação)
    df_cleaned = df_sorted.drop_duplicates(subset=['sinal_id', 'sinalizador_id', 'repeticao_id'], keep='first')
    
    # Remove a coluna 'has_suffix' que não é mais necessária
    df_cleaned = df_cleaned.drop(columns=['has_suffix'])
    # --- Fim da lógica de duplicatas ---

    print(f"\nTotal de vídeos encontrados (antes da limpeza): {len(df)}")
    print(f"Total de vídeos após limpeza de duplicatas: {len(df_cleaned)}")
    print("Exemplo das primeiras 5 linhas do DataFrame (limpo):")
    print(df_cleaned.head())
    
    # Verificações de integridade
    print(f"\nNúmero de sinais únicos: {df_cleaned['sinal_nome'].nunique()}")
    print(f"Sinais únicos: {df_cleaned['sinal_nome'].unique()}")
    print(f"\nNúmero de sinalizadores únicos: {df_cleaned['sinalizador_id'].nunique()}")
    print(f"Sinalizadores únicos: {df_cleaned['sinalizador_id'].unique()}")
    
    # Contagem de amostras por sinal
    print("\nContagem de amostras por sinal (limpo):")
    print(df_cleaned['sinal_nome'].value_counts().sort_index())

    # Contagem de amostras por sinalizador
    print("\nContagem de amostras por sinalizador (limpo):")
    print(df_cleaned['sinalizador_id'].value_counts().sort_index())

    # Verifica os sinais faltantes que você mencionou
    expected_total_samples = 20 * 5 * 12 # 1200
    if len(df_cleaned) < expected_total_samples:
        print(f"\nAviso: Foram encontradas {len(df_cleaned)} amostras, esperava-se {expected_total_samples}.")
        print("Isso pode ser devido aos dados faltantes conforme a documentação.")
        
        missing_info = {
            '03': ['Aluno', 'America', 'Cinco'],
            '04': ['Filho'],
            '09': ['Amarelo', 'Banheiro', 'Conhecer', 'Esquina', 'Medo']
        }
        
        print("\nVerificando ausências esperadas (da documentação):")
        for sig_id, signals in missing_info.items():
            for signal_name in signals:
                # Sinal ID do mapeamento, pois o nome no arquivo pode ser diferente
                signal_id_from_name = next(key for key, value in signal_id_to_name.items() if value == signal_name)
                
                is_missing_in_df = df_cleaned[
                    (df_cleaned['sinalizador_id'] == sig_id) & 
                    (df_cleaned['sinal_nome'] == signal_name)
                ].empty
                
                if is_missing_in_df:
                    print(f"  Confirmado: Sinal '{signal_name}' (ID: {signal_id_from_name}) do Sinalizador {sig_id} está ausente no DataFrame.")
                else:
                    print(f"  Aviso: Sinal '{signal_name}' (ID: {signal_id_from_name}) do Sinalizador {sig_id} FOI ENCONTRADO, mas deveria estar ausente. Verifique manualmente!")

    return df_cleaned

if __name__ == "__main__":
    dataset_base_path = "/data/popo"
    
    metadata_df = extract_video_metadata(dataset_base_path)
    
    output_csv_path = os.path.join(dataset_base_path, "minds_libras_metadata_cleaned.csv") # Novo nome para indicar limpeza
    metadata_df.to_csv(output_csv_path, index=False)
    print(f"\nMetadados limpos salvos em: {output_csv_path}")
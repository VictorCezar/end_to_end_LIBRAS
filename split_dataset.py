import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_dataset(metadata_csv_path, output_dir, test_sinalizadores_ids=None, val_split_ratio=0.15):
    """
    Divide o dataset de metadados em conjuntos de treino, validação e teste.
    A divisão é feita priorizando a separação de sinalizadores completos para o teste.

    Args:
        metadata_csv_path (str): Caminho para o arquivo CSV de metadados limpos.
        output_dir (str): Diretório onde os CSVs de treino, validação e teste serão salvos.
        test_sinalizadores_ids (list, optional): Lista de IDs de sinalizadores (str) para
                                                  serem usados exclusivamente no conjunto de teste.
                                                  Se None, um sinalizador será escolhido aleatoriamente.
        val_split_ratio (float): Proporção dos dados restantes (após separar o teste)
                                 que será usada para o conjunto de validação.

    Returns:
        tuple: (pd.DataFrame, pd.DataFrame, pd.DataFrame) dos conjuntos de treino, validação e teste.
    """
    print(f"Carregando metadados de: {metadata_csv_path}")
    df = pd.read_csv(metadata_csv_path)

    print(f"Total de amostras no DataFrame carregado: {len(df)}")
    print(f"Sinalizadores disponíveis: {df['sinalizador_id'].unique()}")
    print(f"Sinais disponíveis: {df['sinal_nome'].unique()}")

    # Criar mapeamento de rótulos (string -> int)
    # Garante que os IDs dos rótulos sejam consistentes e sequenciais de 0 a N-1
    unique_signals = sorted(df['sinal_nome'].unique())
    label_to_id = {signal: i for i, signal in enumerate(unique_signals)}
    id_to_label = {i: signal for i, signal in enumerate(unique_signals)}

    df['label_id'] = df['sinal_nome'].map(label_to_id)
    print(f"\nMapeamento de Sinais para IDs: {label_to_id}")

    test_df = pd.DataFrame()
    train_val_df = df.copy()

    if test_sinalizadores_ids:
        # Se IDs de sinalizadores para teste foram especificados
        print(f"\nSeparando sinalizadores para o conjunto de TESTE: {test_sinalizadores_ids}")
        
        # Filtra os dados de teste
        test_df = df[df['sinalizador_id'].isin(test_sinalizadores_ids)].copy()
        
        # Os dados restantes são para treino e validação
        train_val_df = df[~df['sinalizador_id'].isin(test_sinalizadores_ids)].copy()
        
        if test_df.empty:
            print("Aviso: Nenhum vídeo encontrado para os sinalizadores de teste especificados. Verifique os IDs.")
            # Se não houver dados de teste, voltamos para a divisão aleatória padrão
            test_sinalizadores_ids = None # Reseta para usar a lógica abaixo

    if test_df.empty: # Caso não tenha sido especificado ou a lista estava vazia/inválida
        print("\nNenhum sinalizador específico para teste definido ou encontrado. Escolhendo um aleatoriamente.")
        all_sinalizadores = df['sinalizador_id'].unique()
        
        # Escolhe um sinalizador aleatoriamente para o teste
        # Use np.random.seed() se quiser reprodutibilidade para essa escolha aleatória
        import numpy as np
        np.random.seed(42) # Para reprodutibilidade
        test_sinalizadores_ids = list(np.random.choice(all_sinalizadores, size=1, replace=False))
        print(f"Sinalizador(es) escolhido(s) aleatoriamente para TESTE: {test_sinalizadores_ids}")

        test_df = df[df['sinalizador_id'].isin(test_sinalizadores_ids)].copy()
        train_val_df = df[~df['sinalizador_id'].isin(test_sinalizadores_ids)].copy()

    # Divisão de Treino e Validação (estratificada por sinal para balanceamento de classes)
    print(f"\nDividindo os {len(train_val_df)} vídeos restantes em Treino e Validação ({100 - val_split_ratio*100:.0f}%/{val_split_ratio*100:.0f}%)...")
    
    # Verifica se há classes suficientes em train_val_df para estratificação
    if train_val_df['label_id'].nunique() < 2:
        print("Aviso: Apenas uma classe de sinal restante para treino/validação. Não é possível estratificar.")
        X_train, X_val = train_test_split(train_val_df, test_size=val_split_ratio, random_state=42)
    else:
        X_train, X_val = train_test_split(train_val_df, test_size=val_split_ratio, random_state=42, stratify=train_val_df['label_id'])

    train_df = X_train.copy()
    val_df = X_val.copy()

    print(f"\n--- Resumo da Divisão ---")
    print(f"Total de amostras: {len(df)}")
    print(f"Amostras de Treino: {len(train_df)}")
    print(f"Amostras de Validação: {len(val_df)}")
    print(f"Amostras de Teste: {len(test_df)}")

    # Verificar se há sobreposição de sinalizadores
    train_sinalizadores = set(train_df['sinalizador_id'].unique())
    val_sinalizadores = set(val_df['sinalizador_id'].unique())
    test_sinalizadores = set(test_df['sinalizador_id'].unique())

    overlap_train_test = train_sinalizadores.intersection(test_sinalizadores)
    overlap_val_test = val_sinalizadores.intersection(test_sinalizadores)
    overlap_train_val = train_sinalizadores.intersection(val_sinalizadores)

    if overlap_train_test:
        print(f"ERRO: Sobreposição de sinalizadores entre Treino e Teste: {overlap_train_test}")
    if overlap_val_test:
        print(f"ERRO: Sobreposição de sinalizadores entre Validação e Teste: {overlap_val_test}")
    if not overlap_train_val:
        print(f"Aviso: Sinalizadores de Treino e Validação não se sobrepõem. Isso é incomum, verifique.")
    else:
        print(f"Sinalizadores de Treino: {sorted(list(train_sinalizadores))}")
        print(f"Sinalizadores de Validação: {sorted(list(val_sinalizadores))}")
        print(f"Sinalizadores de Teste: {sorted(list(test_sinalizadores))}")


    # Salvar os DataFrames divididos
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train_metadata.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val_metadata.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_metadata.csv"), index=False)

    # Salvar o mapeamento de rótulos
    with open(os.path.join(output_dir, "label_mapping.txt"), "w") as f:
        for label, label_id in label_to_id.items():
            f.write(f"{label}:{label_id}\n")
    print(f"Mapeamento de rótulos salvo em: {os.path.join(output_dir, 'label_mapping.txt')}")
    print(f"Conjuntos de dados divididos salvos em: {output_dir}")

    return train_df, val_df, test_df, label_to_id, id_to_label

if __name__ == "__main__":
    # Caminho para o CSV de metadados limpos gerado anteriormente
    cleaned_metadata_csv = "/data/popo/minds_libras_metadata_cleaned.csv"
    
    # Diretório onde os arquivos CSV divididos e o mapeamento serão salvos
    output_splits_dir = "/data/popo/dataset_splits"
    
    # IDs dos sinalizadores para teste.
    # RECOMENDADO: Escolha 1 ou 2 sinalizadores que você não quer que o modelo veja durante o treino/validação.
    # Exemplo: ['11', '12'] para os dois últimos sinalizadores.
    # Se deixar como None, o script escolherá um aleatoriamente para você.
    sinalizadores_para_teste = [11, 12] # Ou None

    train_df, val_df, test_df, label_to_id, id_to_label = split_dataset(
        metadata_csv_path=cleaned_metadata_csv,
        output_dir=output_splits_dir,
        test_sinalizadores_ids=sinalizadores_para_teste
    )

    print("\nDivisão concluída com sucesso!")
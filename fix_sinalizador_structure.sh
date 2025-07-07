#!/bin/bash

# Diretório base onde estão as pastas SinalizadorXX
BASE_DIR="/data/popo"

echo "Padronizando a estrutura de diretórios do dataset MINDS-Libras..."

# Mude para o diretório base
cd "$BASE_DIR" || { echo "Erro: Não foi possível mudar para o diretório $BASE_DIR"; exit 1; }

# Loop através de todas as pastas SinalizadorXX (excluindo a original Sinalizador01 que já está ok no primeiro nível)
# Vamos iterar por cada pasta e verificar a estrutura interna.
for sinalizador_dir in Sinalizador*; do
    if [ -d "$sinalizador_dir" ]; then # Garante que estamos lidando com um diretório
        echo "Verificando diretório: $sinalizador_dir"

        # Caminho onde esperamos encontrar a pasta Canon "um nível abaixo"
        expected_canon_path_deep="$sinalizador_dir/$sinalizador_dir/Canon"
        # Caminho onde a pasta Canon deveria estar no final
        target_canon_path="$sinalizador_dir/Canon"

        # Verifica se a estrutura "SinalizadorXX/SinalizadorXX/Canon" existe e se a pasta "SinalizadorXX/Canon" não existe ainda
        if [ -d "$expected_canon_path_deep" ] && [ ! -d "$target_canon_path" ]; then
            echo "  Estrutura profunda detectada em $sinalizador_dir. Movendo Canon para o nível correto."
            # Mova o conteúdo de SinalizadorXX/SinalizadorXX/Canon para SinalizadorXX/Canon
            mv "$expected_canon_path_deep" "$target_canon_path"
            
            # Remove o diretório SinalizadorXX/SinalizadorXX vazio (se estiver vazio após o mv)
            rmdir "$sinalizador_dir/$sinalizador_dir" 2>/dev/null
            echo "  Movido: $expected_canon_path_deep -> $target_canon_path"
        elif [ -d "$target_canon_path" ]; then
            echo "  Estrutura já padronizada em $sinalizador_dir. Nenhum movimento necessário."
        else
            echo "  Aviso: Estrutura inesperada ou pasta Canon não encontrada no caminho esperado em $sinalizador_dir."
            # Você pode adicionar mais lógica aqui se houver outras variações
        fi
    fi
done

echo "Processo de padronização concluído."
echo "Agora, todos os diretórios deverão ter a estrutura: /data/popo/SinalizadorXX/Canon/"

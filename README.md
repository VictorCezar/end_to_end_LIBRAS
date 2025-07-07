# Classificador de Sinais de LIBRAS com PyTorch e Transformers

## 📖 Descrição

Este projeto é uma implementação de um modelo de Deep Learning para o reconhecimento e classificação de sinais isolados da Língua Brasileira de Sinais (LIBRAS) a partir de vídeos. Utilizando uma arquitetura *end-to-end*, o modelo analisa diretamente os pixels do vídeo para realizar a classificação, com foco em alta acurácia e eficiência computacional.

O trabalho foi desenvolvido como parte de um artigo científico, comparando diferentes arquiteturas de redes neurais, como 3D-CNNs (I3D) e Video Transformers.

### ✨ Features

* **Alta Acurácia:** Atinge mais de 90% de acurácia no conjunto de teste.
* **Pipeline End-to-End:** Processa diretamente vídeos brutos (RGB).
* **Validação Robusta:** O conjunto de teste é composto por sinalizadores não vistos durante o treinamento, garantindo a capacidade de generalização do modelo.
* **Reprodutibilidade:** O projeto é totalmente containerizado com Docker, garantindo que qualquer pessoa possa rodar o treinamento com dois comandos.

## 🛠️ Tecnologias Utilizadas

* Python 3.10+
* PyTorch
* FFmpeg
* Conda
* Docker
* NVIDIA CUDA

## ⚙️ Estrutura do Projeto

```
.
├── dataset_splits/     # Arquivos CSV com a divisão de treino, validação e teste
├── training_plots/       # Pasta onde os gráficos de treinamento são salvos
├── build_pytorch_model.py # Script principal para treinamento e avaliação do modelo
├── extract_metadata.py    # Script para extrair metadados dos vídeos
├── split_dataset.py       # Script para realizar a divisão do dataset
├── minds_libras_metadata.csv # Metadados do dataset
├── Dockerfile             # Receita para construir a imagem Docker do projeto
├── environment.yml        # Lista de dependências do ambiente Conda
└── README.md              # Esta documentação
```

## 🚀 Instalação e Uso

Você pode rodar este projeto de duas maneiras: a forma recomendada (com Docker) ou a forma manual.

### Método 1: Usando Docker (Recomendado)

**Pré-requisitos:** [Docker](https://www.docker.com/get-started) e [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (para suporte a GPU).

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)
    cd seu-repositorio
    ```

2.  **Construa a imagem Docker:**
    ```bash
    docker build -t libras-classifier .
    ```

3.  **Execute o treinamento:**
    ```bash
    docker run --gpus all -v "$(pwd)/dataset_splits:/app/dataset_splits" -v "$(pwd)/training_plots:/app/training_plots" libras-classifier
    ```
    Os gráficos e resultados do treinamento aparecerão na pasta `training_plots`.

### Método 2: Manualmente com Conda

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)
    cd seu-repositorio
    ```

2.  **Crie o ambiente Conda a partir do arquivo de ambiente:**
    ```bash
    conda env create -f environment.yml
    ```

3.  **Ative o ambiente:**
    ```bash
    conda activate libras_env  # Substitua 'libras_env' pelo nome do seu ambiente
    ```

4.  **Execute o script de treinamento:**
    ```bash
    python build_pytorch_model.py
    ```

## 🤝 Contribuições

Contribuições, issues e feature requests são bem-vindos! Sinta-se à vontade para abrir uma issue para discutir melhorias.
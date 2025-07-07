# 1. Imagem Base
# Imagem oficial do PyTorch com suporte a CUDA 12.1.
# Isso economiza muito tempo, pois já vem com Python, Conda, PyTorch e drivers NVIDIA.
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# 2. Definir o Diretório de Trabalho
# Define o diretório padrão dentro do container.
WORKDIR /app

# 3. Copiar o arquivo de dependências e criar o ambiente Conda
# Copiamos primeiro para aproveitar o cache do Docker. Se o environment.yml não mudar,
# o Docker não vai reinstalar tudo de novo a cada build.
COPY environment.yml .
RUN conda env create -f environment.yml

# 4. Configurar o Shell para usar o ambiente Conda por padrão
# Substitua 'libras_env' pelo nome do seu ambiente (verifique a primeira linha do environment.yml)
SHELL ["conda", "run", "-n", "libras_env", "/bin/bash", "-c"]

# 5. Copiar o restante do código do projeto para o container
COPY . .

# 6. Comando Padrão
# Comando que será executado quando o container iniciar.
# Ele roda o script principal de treinamento do modelo.
CMD ["python", "build_pytorch_model.py"]
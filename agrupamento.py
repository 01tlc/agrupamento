import os
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.applications import InceptionResNetV2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Caminhos do dataset e dos pesos do modelo
dataset_path = "./dataset_faces"  # Altere conforme necessário
weights_path = "./facenet_keras.h5"  # Altere conforme necessário

# Função para carregar o modelo FaceNet
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Input

def load_facenet_model(weights_path=None):
    """
    Carrega o modelo FaceNet com pesos do ImageNet.
    """
    print("Carregando modelo FaceNet com pesos do ImageNet...")
    input_tensor = Input(shape=(160, 160, 3))
    model = InceptionResNetV2(include_top=False, weights="imagenet", input_tensor=input_tensor, pooling="avg")
    print("Modelo FaceNet carregado com sucesso (ImageNet).")
    return model

# Função para carregar e processar imagens
def load_and_preprocess_images(dataset_path):
    """
    Carrega e processa imagens do dataset para o formato esperado pelo modelo.
    """
    print("Carregando e processando imagens...")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Caminho do dataset não encontrado: {dataset_path}")

    images = []
    file_paths = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith((".jpg", ".png")):
                file_path = os.path.join(root, file)
                image = cv2.imread(file_path)
                if image is None:
                    print(f"Falha ao carregar a imagem: {file_path}")
                    continue
                image = cv2.resize(image, (160, 160))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0  # Normaliza para [0, 1]
                images.append(image)
                file_paths.append(file_path)
    print(f"{len(images)} imagens carregadas.")
    return np.array(images), file_paths

# Função para extrair embeddings
def extract_embeddings(images, model):
    """
    Extrai os embeddings faciais usando o modelo FaceNet.
    """
    print("Extraindo embeddings faciais...")
    embeddings = []
    for i, image in enumerate(images):
        try:
            embedding = model.predict(np.expand_dims(image, axis=0))[0]
            embeddings.append(embedding)
        except Exception as e:
            print(f"Erro ao processar a imagem {i}: {e}")
            embeddings.append(None)
    valid_embeddings = [emb for emb in embeddings if emb is not None]
    print(f"{len(valid_embeddings)} embeddings válidos extraídos.")
    return np.array(valid_embeddings)

# Função para clusterizar embeddings
def cluster_embeddings(embeddings):
    """
    Realiza a clusterização dos embeddings usando DBSCAN.
    """
    print("Realizando clusterização com DBSCAN...")
    if embeddings.shape[0] == 0:
        print("Nenhum embedding disponível para clusterização.")
        return []
    embeddings = PCA(n_components=40).fit_transform(embeddings)  # Reduz dimensionalidade
    clustering_model = DBSCAN(eps=0.5, min_samples=2, metric="cosine")
    labels = clustering_model.fit_predict(embeddings)
    print(f"Clusters identificados: {len(set(labels)) - (1 if -1 in labels else 0)}")
    return labels

# Função para visualizar clusters
def visualize_clusters(labels, file_paths):
    """
    Exibe os clusters e os caminhos das imagens pertencentes a cada um.
    """
    unique_labels = set(labels)
    for label in unique_labels:
        print(f"Cluster {label}:")
        cluster_images = [file_paths[i] for i in range(len(labels)) if labels[i] == label]
        for img_path in cluster_images:
            print(f"  - {img_path}")
        print()

# Função para mostrar imagens dos clusters
def show_cluster_images(labels, images):
    """
    Exibe visualmente as imagens agrupadas em clusters.
    """
    unique_labels = set(labels)
    for label in unique_labels:
        cluster_images = [images[i] for i in range(len(labels)) if labels[i] == label]
        plt.figure(figsize=(10, 5))
        for i, img in enumerate(cluster_images[:10]):  # Mostra no máximo 10 imagens por cluster
            plt.subplot(2, 5, i + 1)
            plt.imshow(img)
            plt.axis("off")
        plt.suptitle(f"Cluster {label}")
        plt.show()

# Pipeline principal
def main():
    try:
        # Carregar o modelo FaceNet
        model = load_facenet_model(weights_path)

        # Carregar e processar imagens
        images, file_paths = load_and_preprocess_images(dataset_path)

        # Extrair embeddings
        embeddings = extract_embeddings(images, model)

        # Realizar clusterização
        if len(embeddings) > 0:
            labels = cluster_embeddings(embeddings)

            # Visualizar os clusters
            visualize_clusters(labels, file_paths)
            show_cluster_images(labels, images)
        else:
            print("Nenhum embedding foi extraído. Verifique os dados de entrada.")
    except Exception as e:
        print(f"Erro na execução do pipeline: {e}")

if __name__ == "__main__":
    main()

import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from keras_facenet import FaceNet
import random

# Path to dataset
dataset_path = "./dataset_faces"  # Update with your dataset path


# Load FaceNet model
def load_facenet_model():
    """
    Load the FaceNet model for embedding extraction.
    """
    print("Loading FaceNet model...")
    embedder = FaceNet()  # This uses a compatible FaceNet model
    print("FaceNet model loaded successfully.")
    return embedder.model


# Load and preprocess images
def load_and_preprocess_images(dataset_path):
    """
    Load and preprocess images for the model.
    """
    print("Loading and preprocessing images...")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    images = []
    file_paths = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith((".jpg", ".png")):
                file_path = os.path.join(root, file)
                image = cv2.imread(file_path)
                if image is None:
                    print(f"Failed to load image: {file_path}")
                    continue
                image = cv2.resize(image, (160, 160))  # FaceNet input size
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0  # Normalize to [0, 1]
                images.append(image)
                file_paths.append(file_path)
    print(f"{len(images)} images loaded and preprocessed.")
    return np.array(images), file_paths


# Extract embeddings
def extract_embeddings(images, model):
    """
    Extract facial embeddings from images using the FaceNet model.
    """
    print("Extracting facial embeddings...")
    embeddings = []
    for i, image in enumerate(images):
        try:
            embedding = model.predict(np.expand_dims(image, axis=0))[0]
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error processing image {i}: {e}")
            embeddings.append(None)
    valid_embeddings = [emb for emb in embeddings if emb is not None]
    print(f"{len(valid_embeddings)} valid embeddings extracted.")
    return np.array(valid_embeddings)


# Perform clustering with K-Means
def cluster_embeddings_kmeans(embeddings, n_clusters=10):
    """
    Cluster facial embeddings using K-Means clustering.
    """
    print("Performing K-Means clustering...")
    if len(embeddings) == 0:
        print("No embeddings available for clustering.")
        return []
    
    # Normalize embeddings
    embeddings = normalize(embeddings)
    
    # Dimensionality reduction for clustering
    embeddings = PCA(n_components=40).fit_transform(embeddings)
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    
    print(f"Clusters created: {n_clusters}")
    return labels


# Enforce exactly 4 images per cluster
def enforce_fixed_cluster_size(labels, file_paths, images, fixed_size=4):
    """
    Ensure each cluster contains exactly 'fixed_size' images.
    """
    print("Enforcing fixed cluster size...")
    unique_labels = set(labels)
    new_labels = []
    new_images = []
    new_file_paths = []
    
    for label in unique_labels:
        # Find indices of images in this cluster
        indices = [i for i, lbl in enumerate(labels) if lbl == label]
        
        # Randomly select `fixed_size` images from this cluster
        if len(indices) >= fixed_size:
            selected_indices = random.sample(indices, fixed_size)
        else:
            print(f"Cluster {label} has fewer than {fixed_size} images. Adjusting.")
            selected_indices = indices  # Keep fewer images if the cluster is small
        
        # Add selected images and file paths
        new_labels.extend([label] * len(selected_indices))
        new_images.extend([images[i] for i in selected_indices])
        new_file_paths.extend([file_paths[i] for i in selected_indices])
    
    return new_labels, new_file_paths, new_images


# Visualize clusters
def visualize_clusters(labels, file_paths):
    """
    Display the clusters and the file paths of images in each cluster.
    """
    unique_labels = set(labels)
    for label in unique_labels:
        print(f"Cluster {label}:")
        cluster_images = [file_paths[i] for i in range(len(labels)) if labels[i] == label]
        for img_path in cluster_images:
            print(f"  - {img_path}")
        print()


# Show cluster images
def show_cluster_images(labels, images):
    """
    Visually display images grouped by clusters.
    """
    unique_labels = set(labels)
    for label in unique_labels:
        cluster_images = [images[i] for i in range(len(labels)) if labels[i] == label]
        plt.figure(figsize=(10, 5))
        for i, img in enumerate(cluster_images[:10]):  # Show max 10 images per cluster
            plt.subplot(2, 5, i + 1)
            plt.imshow(img)
            plt.axis("off")
        plt.suptitle(f"Cluster {label}")
        plt.show()


# Main pipeline
def main():
    try:
        # Load FaceNet model
        model = load_facenet_model()

        # Load and preprocess images
        images, file_paths = load_and_preprocess_images(dataset_path)

        # Extract embeddings
        embeddings = extract_embeddings(images, model)

        # Perform clustering
        if len(embeddings) > 0:
            labels = cluster_embeddings_kmeans(embeddings, n_clusters=10)  # Specify 10 clusters
            
            # Enforce exactly 4 images per cluster
            labels, file_paths, images = enforce_fixed_cluster_size(labels, file_paths, images, fixed_size=4)

            # Visualize and show clusters
            visualize_clusters(labels, file_paths)
            show_cluster_images(labels, images)
        else:
            print("No embeddings extracted. Check input data.")
    except Exception as e:
        print(f"Pipeline execution error: {e}")


# Entry point
if __name__ == "__main__":
    main()

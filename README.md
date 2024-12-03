Ativdade: Agrupamento Facial 

Desafio: 

Desenvolva uma aplicação que leia o dataset de rostos (em anexo) e realize o agrupamento 
das imagens por semelhança, de modo que cada pessoa no dataset tenha suas respectivas 
imagens agrupadas. A aplicação deve usar técnicas de aprendizado de máquina e/ou 
processamento de imagem para identificar e agrupar rostos semelhantes. 

Requisitos: 
Leitura do Dataset: A aplicação deve ser capaz de carregar e ler um dataset de imagens de 
rostos de forma aleatória, garantindo que a ordem de leitura das imagens não interfira no 
processo. 

Agrupamento de Semelhança: Implemente um algoritmo de agrupamento (clustering) 
para agrupar as imagens de acordo com a semelhança dos rostos. 

Visualização dos Resultados: A aplicação deve exibir os grupos formados, mostrando 
todas as imagens pertencentes a cada grupo. 
Critérios de Avaliação: 

Precisão do Agrupamento: O quão bem as imagens foram agrupadas de acordo com a 
semelhança dos rostos. 

Qualidade do Código: Organização, clareza e manutenibilidade do código. 
Documentação: A aplicação deve incluir uma documentação clara sobre como configurar 
e executar o projeto. 


Sugestões de Ferramentas e Bibliotecas: 
• OpenCV 
• scikit-learn 
• dlib 
• facenet 
• pytorch 
• numpy 

Entrega: Os candidatos devem submeter o código-fonte completo, incluindo um arquivo 
README.md com instruções de instalação e uso, e exemplos de saída da aplicação. 
###############################################################################################################################################

Face Clustering Challenge
Overview
This project implements a face clustering solution using machine learning techniques. The goal is to group a set of face images into clusters based on similarity, ensuring that 40 images are grouped into 10 clusters, with exactly 4 images per cluster.

The solution leverages the FaceNet model for facial embedding extraction and K-Means clustering to group the embeddings. Post-processing ensures the cluster sizes are fixed to 4 images.

Challenge Requirements
Objective
Develop an application that:

Reads a dataset of face images.
Clusters the images into groups based on facial similarity.
Displays the clusters with all images grouped appropriately.
Specific Requirements
Input: A dataset of face images (e.g., .jpg, .png).
Output: 10 clusters, with each cluster containing exactly 4 images.
Machine Learning Techniques: Use face recognition and clustering algorithms to group images based on similarity.
Visualization: Display the images grouped into clusters for review.
Solution Design
Steps in the Pipeline
Data Loading and Preprocessing:

The dataset is loaded from a user-defined path.
Each image is resized to 160x160 (FaceNet input size) and normalized.
Facial Embedding Extraction:

The FaceNet model is used to extract facial embeddings for each image. These embeddings are numerical representations of facial features.
Clustering with K-Means:

The embeddings are clustered into exactly 10 clusters using the K-Means clustering algorithm.
Enforcing Fixed Cluster Size:

Post-processing ensures that each cluster contains exactly 4 images. If a cluster has more than 4 images, a random sample of 4 is taken. If fewer, all images in the cluster are used.
Visualization:

The images are displayed grouped by their respective clusters.
File paths for each cluster are printed to the console for reference.
Code Details
Key Components
FaceNet Model:
Pre-trained FaceNet embeddings are extracted using the keras-facenet library.
Clustering Algorithm:
K-Means: Groups embeddings into exactly 10 clusters.
Fixed Cluster Size: Ensures each cluster contains exactly 4 images.
Visualization:
Clusters are displayed both textually and visually.
Directory Structure
project/
│
├── dataset_faces/        # Directory containing the face images
├── script.py             # Main Python script
└── README.md             # This documentation
Key Functions
load_facenet_model: Loads the pre-trained FaceNet model.
load_and_preprocess_images: Loads images from the dataset and preprocesses them for embedding extraction.
extract_embeddings: Extracts embeddings for each image using the FaceNet model.
cluster_embeddings_kmeans: Clusters embeddings into 10 groups using K-Means.
enforce_fixed_cluster_size: Ensures each cluster contains exactly 4 images.
visualize_clusters and show_cluster_images: Displays file paths and visualizes images in clusters.
Installation and Usage
Prerequisites
Ensure you have Python installed along with the following libraries:

keras-facenet
opencv-python
numpy
scikit-learn
matplotlib
Install the dependencies:

pip install keras-facenet opencv-python numpy scikit-learn matplotlib
Running the Solution
Prepare the Dataset:

Place your face images in the dataset_faces directory. Ensure the directory contains 40 images in .jpg or .png format.
Run the Script:

python script.py
View the Results:

File paths for each cluster are printed to the console.
Visualizations of the clusters are displayed, with 4 images per cluster.
Output Details
Console Output
The script prints file paths for each cluster, for example:

Cluster 0:
  - dataset_faces/image1.jpg
  - dataset_faces/image2.jpg
  - dataset_faces/image3.jpg
  - dataset_faces/image4.jpg
...
Visual Output
Images are grouped and displayed in clusters, with 4 images shown per cluster.

Customization
Adjusting the Number of Clusters or Images per Cluster
Number of Clusters: Modify the n_clusters parameter in the cluster_embeddings_kmeans function.
Images per Cluster: Change the fixed_size parameter in the enforce_fixed_cluster_size function.
Limitations and Future Improvements
Dynamic Cluster Sizes: Current implementation fixes each cluster to 4 images. A future enhancement could allow dynamic sizing based on similarity scores.
Face Alignment: Adding a preprocessing step for face alignment can improve clustering accuracy.
Alternate Clustering Algorithms: Experimenting with algorithms like Agglomerative Clustering or Gaussian Mixture Models may improve performance for larger datasets.
Example Results
Input Dataset
40 face images, with varying similarities.
Output
10 Clusters with exactly 4 images per cluster.
Visualizations of clusters for easy interpretation.
Contact
For questions or feedback, please contact [Your Name/Email].

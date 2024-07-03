import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score
import os 
import shutil
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics.pairwise import pairwise_distances
from tkinter import filedialog
from scipy.spatial import ConvexHull
import sys
from tqdm import tqdm

 
def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.Grayscale(num_output_channels=3), 
        transforms.ToTensor(),
    ])
    
    img = Image.open(img_path)
    if img.mode != 'L':
        img = img.convert('L')
    img = transform(img)
    img = img.unsqueeze(0)
    
    return img


def hook_fn(module, input, output):
    intermediate_features.append(output)


def extract_features(model, img, layer_index): 
    """
    Extractes features from certain layer in model 
    """

    global intermediate_features
    intermediate_features = []
    hook = model.model.model[layer_index].register_forward_hook(hook_fn)

    with torch.no_grad(): # not calculating forward pass
        model(img)

    hook.remove()
    if intermediate_features:
        
        return intermediate_features[0]
    else:
        return None
    

def create_features_list(model, image_paths, layer_index, sample_percentage):  
    features_dict = {}

    random.shuffle(image_paths)
    selected_paths = image_paths[:int(len(image_paths) * sample_percentage)]
    for i in tqdm(range(len(selected_paths))):
        original_stdout = sys.stdout

# Redirect stdout to /dev/null or equivalent
        sys.stdout = open(os.devnull, 'w')
        img_path =  selected_paths[i]
 
        img = preprocess_image(img_path)
    
        features = extract_features(model, img, layer_index)

        if features is not None:
            
            features_flattened = features.view(features.size(0), -1)
            features_dict[img_path.name] = features_flattened.cpu().numpy()

        sys.stdout = original_stdout

    return features_dict


def plot_features_space(features_dict):
    """
    Plots featured space flattened to 2D space 
    """
    X_values = np.array(list(features_dict.values())) 

    # Reshape the array to flatten the extra dimension
    X_values = X_values.reshape(X_values.shape[0], -1)

    # Standardize the features
    scaler = StandardScaler()
    X_values = scaler.fit_transform(X_values)

    plt.scatter(X_values[:, 0], X_values[:, 1])
    plt.title("Data before t-SNE")
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()


def applying_tsne(features, perplexity):
    """
    Applying t-SNE algorithm to featured data, transforming into a 2D space 
    """
    X_values = np.array(list(features.values())) 

    if len(X_values) == 0:
        print("No features detected.")
        return None
    X_values = X_values.reshape(len(X_values), -1)
    X_keys = np.array(list(features.keys()))
    X_values = StandardScaler().fit_transform(X_values)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    coordinates_tsne = tsne.fit_transform(X_values)
    X_tsne_with_names = np.column_stack((X_keys, coordinates_tsne))

    return X_tsne_with_names


def plot_reduced_feature_space(features_tsne):
    """
    Plotting the feature space after t-SNE has been applied
    """

    coordinates_tsne = features_tsne[:, 1:].astype(float)
    plt.scatter(coordinates_tsne[:, 0], coordinates_tsne[:, 1])
    plt.title('Data after t-SNE')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()


def cluster_and_plot(features_tsne, epsilon, min_samples, image_paths, image_dir):

    coordinates_tsne = features_tsne[:, 1:].astype(float)
    db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(coordinates_tsne)
    labels = db.labels_

    # db_index = davies_bouldin_score(coordinates_tsne, labels)
    # print("Davies-Bouldin Index: ", db_index)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print("Estimated number of clusters: %d" % n_clusters)
    print("Estimated number of noise points: %d" % n_noise)


    plt.scatter(coordinates_tsne[:, 0], coordinates_tsne[:, 1], c=labels, cmap='viridis')
    cluster_centers = {}
   
 
    clusters_dir = f"{image_dir.parent}/clusters/"
    shutil.rmtree(clusters_dir, ignore_errors=True)
    os.makedirs(clusters_dir, exist_ok=True)
    for cluster_label in set(labels):
        if cluster_label != -1:  # Ignore noise points
            cluster_points = coordinates_tsne[labels == cluster_label]
            cluster_indices = np.where(labels == cluster_label)[0]
            cluster_center = np.mean(cluster_points, axis=0)
            cluster_centers[cluster_label] = cluster_center
            plt.text(cluster_center[0], cluster_center[1], str(cluster_label),
                    color='red', fontsize=12, weight='bold')
            
            all_hull_indices = []
            simplices = []
            num_hulls = len(cluster_points) // 100
            num_hulls = max(3, num_hulls)
            for _ in range(num_hulls):

                if len(cluster_points) < 3:
                    break
                hull = ConvexHull(cluster_points)
                hull_indices = hull.vertices
                all_hull_indices.append(hull_indices)
                for simplex in hull.simplices:
              
        
                    simplices.append((cluster_points[simplex, 0], cluster_points[simplex, 1]))
                cluster_points = np.delete(cluster_points, hull_indices, axis=0)
                cluster_indices = np.delete(cluster_indices, hull_indices, axis=0)
            for x, y in simplices:
                plt.plot(x, y, 'k-', lw=1)
           
            centroid_dir = os.path.join(clusters_dir, f"cluster_{cluster_label}/")
            os.makedirs(centroid_dir, exist_ok=True)

            for idxs in all_hull_indices:
                for idx in idxs:
                    image_path = features_tsne[idx][0]
                    shutil.copy(f"{image_dir.parent}/extracted_frames/{image_path}", centroid_dir)
  

    plt.title('DBSCAN Clustering after t-SNE with Convex Hulls')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plot_save_path = os.path.join(f"{image_dir.parent}\\", "cluster_plot.jpg")
    plt.savefig(plot_save_path, format='jpg', dpi=300)
    plt.show()
  
    
    return cluster_centers


def export_features(features_dict, filename):
    np.save(filename, features_dict)


def initialize_clustering(image_dir, model_path):
    global IMAGE_PATHS
    # MODEL_PATH = filedialog.askopenfilename(initialdir="/", title="SELECT MODEL/WEIGHTS FILE")
    # MODEL_PATH = Path("runs/detect/train37/weights/best.pt")

    # IMAGE_FOLDER = filedialog.askdirectory(initialdir="/", title="SELECT IMAGE FOLDER")
    # IMAGE_DIR = Path(IMAGE_FOLDER)
 
    # INCORRECT_IMAGE_DIR = Path("incorrectly_detected")
    # INCORRECT_IMAGE_PATHS = sorted(INCORRECT_IMAGE_DIR.glob("*.png"))
    IMAGE_PATHS = sorted(image_dir.glob("*.jpg")) # change to *.jpg for jpg images
    PERPLEXITY = 10 * (len(IMAGE_PATHS))/1000
    


    LAYER_INDEX = 21 # 21 is the last index before detection head, choosing highest layer for high level features 
    # 20 works and 22? 
    SAMPLE_PERCANTAGE = 1

    #IMAGES_PER_CLUSTERS = 5

    # t-SNE params
    # GOOD PARAMS: (~1000 images: (15, 5), (20, 5)), (~2000 images: (30, 5), )
    # new dataaset ~900 (20, 5), ~ 1800 (20, 5)
    #PERPLEXITY = 10 # controls number of neighbors used in the algorithm, a guess about the number of close neigbors each point has

    # It determines the balance between local and global aspects of the data manifold. Higher perplexity values tend to result in more global views of the data, while lower perplexity values focus more on local structure.

    # Low values focus on local structure, emphasizing smaller-scale structures in the data. Lead to tighter clusters 

    # High values result in a more global view of the data, leading to more spread-out clusters in the visualization and preserving the global structure of the data

    # DBScan params
    EPSILON = 5 # distance measure to locate points in the neighborhood of any point 
    MIN_SAMPLES = 10 # number of points clustered together for a region to be considered dense 


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = YOLO(model_path)
    model.to(device)

    print("Grabbing image features.....")
    features = create_features_list(model, IMAGE_PATHS, LAYER_INDEX, SAMPLE_PERCANTAGE)

    # print("Plotting image features")
    # plot_features_space(features)

    print("number of images:", int(len(IMAGE_PATHS) * SAMPLE_PERCANTAGE))
    print("Applying t-SNE.....")
    features_tsne = applying_tsne(features, PERPLEXITY)

    print("Plotting transformed feature space.....")
    plot_reduced_feature_space(features_tsne)

    cluster_centers = cluster_and_plot(features_tsne, EPSILON, MIN_SAMPLES, IMAGE_PATHS, image_dir)

    export_features(features, f"{image_dir.parent}/features_list.npy")
    # incorrect_img_features = create_features_list(model, INCORRECT_IMAGE_PATHS, LAYER_INDEX, sample_percentage=0.15)
    # print("Matching images to clusters.....")

    # incorrect_features_tsne = applying_tsne(incorrect_img_features, PERPLEXITY)
    
    # print(incorrect_features_tsne)
    # print(cluster_centers)


    # distances_to_clusters = pairwise_distances(incorrect_features_tsne[:, 1:], cluster_centers)


    # farthest_cluster_indices = np.argmin(distances_to_clusters, axis=1)


    # cluster_labels = [f"cluster_{idx}" for idx in farthest_cluster_indices]

    # cluster_image_dict = {cluster_label: [] for cluster_label in set(cluster_labels)}


    # for img_path, cluster_label in zip(INCORRECT_IMAGE_PATHS, cluster_labels):
    #     cluster_image_dict[cluster_label].append(img_path)


    # for cluster_label, image_paths in cluster_image_dict.items():
    #     print(f"Images in {cluster_label}:")
    #     for img_path in image_paths:
    #         print(img_path.name)

    # for cluster_label, image_paths in cluster_image_dict.items():
    
    #     fig, axes = plt.subplots(1, IMAGES_PER_CLUSTERS, figsize=(15, 3))
    #     fig.suptitle(f"Images in {cluster_label}", fontsize=16)


    #     for i in range(IMAGES_PER_CLUSTERS):
    #         image = plt.imread(image_paths[i])
    #         axes[i].imshow(image, cmap="gray")
    #         axes[i].axis('off')
    #         axes[i].set_title(image_paths[i].name)  # Set title as the image filename

    #     plt.show()       
if __name__ == "__main__":
    main()
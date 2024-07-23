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
from tkinter import filedialog
from scipy.spatial import ConvexHull
import sys
from tqdm import tqdm

 
def preprocess_image(img_path):
    """
    Preprocess the image for model input

    Params:
        img_path (str): path to image

    Returns:
        torch.Tensor: Preprocessed image tensor
    """
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
    """
    Hook function to capture intermediate features from a specified layer

    Parameters:
        module (torch.nn.Module): the layer module
        input (tuple): input to the layer
        output (torch.Tensor): output from the layer
    """

    intermediate_features.append(output)


def extract_features(model, img, layer_index): 
    """
    Extract features from a specified layer in the model
    
    Parameters:
        model (torch.nn.Module): the model from which to extract features
        img (torch.Tensor): the input image tensor
        layer_index (int): index of the layer to extract features from
    
    Returns:
        torch.Tensor or None: The extracted features or None if no features were extracted.
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
    """
    Create a dictionary of image features extracted from a specified layer of the model
    
    Parameters:
        model (torch.nn.Module): the model from which to extract features
        image_paths (list): list of paths to the images
        layer_index (int): index of the layer to extract features from
        sample_percentage (float): percentage of images to sample for feature extraction
    
    Returns:
        features_dict (dict_): a dictionary where keys are image file names and values are flattened feature arrays
    """

    features_dict = {}

    random.shuffle(image_paths)
    print(image_paths)
    selected_paths = image_paths[:int(len(image_paths) * sample_percentage)]
    for i in tqdm(range(len(selected_paths))):
        original_stdout = sys.stdout

# Redirect stdout to /dev/null or equivalent
        sys.stdout = open(os.devnull, 'w')
        img_path =  selected_paths[i]
        print(img_path, "img)path")
        img = preprocess_image(img_path)
    
        features = extract_features(model, img, layer_index)

        if features is not None:
            
            features_flattened = features.view(features.size(0), -1)
            features_dict[img_path.name] = features_flattened.cpu().numpy()

        sys.stdout = original_stdout

    return features_dict


def plot_features_space(features_dict):
    """
    Plot the feature space flattened to 2D space
    
    Parameters:
        features_dict (dict): a dictionary where keys are image file names and values are feature arrays
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
    Apply t-SNE algorithm to feature data, transforming it into a 2D space
    
    Parameters:
        features (dict): A dictionary where keys are image file names and values are feature arrays
        perplexity (int): The perplexity parameter for t-SNE
    
    Returns:
        np.ndarray: A numpy array with image file names and their corresponding 2D t-SNE coordinates
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
    Plot the feature space after t-SNE has been applied.
    
    Params:
        features_tsne (np.ndarray): A numpy array with image file names and their corresponding 2D t-SNE coordinates.
    """

    coordinates_tsne = features_tsne[:, 1:].astype(float)
    plt.scatter(coordinates_tsne[:, 0], coordinates_tsne[:, 1])
    plt.title('Data after t-SNE')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()


def cluster_and_plot(features_tsne, epsilon, min_samples, image_dir):
    """

    Params:
        feautures_tsne:
        epsilon (int):
        min_samples (int): 
    """
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
    """
    Export features to a .npy file.
    
    Parameters:
        features_dict (dict): A dictionary where keys are image file names and values are feature arrays
        filename (str): The name of the file to save the features
    """
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
    print(IMAGE_PATHS)

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


    device = torch.device('cpu')
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
    
if __name__ == "__main__":
    #image_dir = Path("used_videos\\f_2024_03_14_13_12_21_03\extracted_frames")
    #model_path = "..\good.pt"
    image_dir = ""
    model_path = ""
    initialize_clustering(image_dir, model_path)
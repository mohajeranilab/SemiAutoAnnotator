import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from ultralytics import YOLO, RTDETR
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import os 
import shutil
from scipy.spatial import ConvexHull
from tqdm import tqdm
from math import ceil
 
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

    # check if it grayscale 
    if img.mode != 'L':
        img = img.convert('L')
    img = transform(img)
    img = img.unsqueeze(0)
    
    return img


def hook_fn(module, input, output):
    """
    Hook function to capture intermediate features from a specified layer

    Params:
        module (torch.nn.Module): the layer module
        input (tuple): input to the layer
        output (torch.Tensor): output from the layer
    """
    global intermediate_features

    intermediate_features.append(output)


def extract_features(model, img, layer_index): 
    """
    Extract features from a specified layer in the model
    
    Params:
        model (torch.nn.Module): the model from which to extract features
        img (torch.Tensor): the input image tensor
        layer_index (int): index of the layer to extract features from
    
    Returns:
        torch.Tensor or None: The extracted features or None if no features were extracted.
    """

    global intermediate_features
    intermediate_features = []
  
    # register a forward hook to the specified layer in the model
   
    hook = model.model.model[layer_index].register_forward_hook(hook_fn)

    # perform a forward pass through the model without gradients
    with torch.no_grad():
        try:
            model(img)
        except KeyError as e:
            print(f"KeyError: {e}")
            print(f"Model names: {model.names}")
            raise
    
    # remove the hook after forward pass
    hook.remove()

    # return the captured intermediate features if available
    if intermediate_features:    
        return intermediate_features[0]
    else:
        return None
   

def create_features_list(model, image_paths, layer_index, sample_percentage):  
    """
    Create a dictionary of image features extracted from a specified layer of the model
    
    Params:
        model (torch.nn.Module): the model from which to extract features
        image_paths (list): list of paths to the images
        layer_index (int): index of the layer to extract features from
        sample_percentage (float): percentage of images to sample for feature extraction
    
    Returns:
        features_dict (dict_): a dictionary where keys are image file names and values are flattened feature arrays
    """

    features_dict = {}

    random.shuffle(image_paths)

    selected_paths = image_paths[:int(len(image_paths) * sample_percentage)]

    # draw a tqdm progress line to see how many images have their features extracted
    for i in tqdm(range(len(selected_paths))):
        img_path = selected_paths[i]

        img = preprocess_image(img_path)
    
        features = extract_features(model, img, layer_index)

        if features is not None:
            features_dict[img_path] = features[0].cpu().numpy()
            
    return features_dict


def plot_features_space(features_dict):
    """
    Plot the feature space flattened to 2D space
    
    Params:
        features_dict (dict): a dictionary where keys are image file names and values are feature arrays
    """

    X_values = np.array(list(features_dict.values())) 

    # reshape the array to flatten the extra dimension
    X_values = X_values.reshape(X_values.shape[0], -1)

    
    X_values = StandardScaler.fit_transform(X_values)

    plt.scatter(X_values[:, 0], X_values[:, 1])
    plt.title("Data before t-SNE")
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()


def applying_tsne(features, perplexity):
    """
    Apply t-SNE algorithm to feature space to reduce it to 2D.
    
    Params:
        features (dict): A dictionary where keys are image file names and values are feature arrays
        perplexity (int): controls the effective number of neighbors that each point considers during dimensionality reduction
    
    Returns:
        np.ndarray: A numpy array with image file names and their corresponding 2D t-SNE coordinates
    """

    X_values = np.array(list(features.values())) 

    if len(X_values) == 0:
        print("No features detected.")
        return None

    X_values = X_values.reshape(len(X_values), -1)

    X_keys = np.array(list(features.keys()))

    # scale the features
    X_values = StandardScaler().fit_transform(X_values)

    # create t-SNE transform
    tsne = TSNE(n_components=2, perplexity=perplexity)
    coordinates_tsne = tsne.fit_transform(X_values)

    # write back img names with their t-SNE features
    X_tsne_with_names = np.column_stack((X_keys, coordinates_tsne))

    return X_tsne_with_names


def plot_reduced_feature_space(features_tsne):
    """
    Plot the feature space after t-SNE has been applied.
    
    Params:
        features_tsne (np.ndarray): A numpy array with image file names and their corresponding 2D coordinates.
    """

    coordinates_tsne = features_tsne[:, 1:].astype(float)
    plt.scatter(coordinates_tsne[:, 0], coordinates_tsne[:, 1])
    plt.title('Data after t-SNE')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()


def cluster_and_plot(features_tsne, epsilon, min_samples, image_dir):
    """
    Perform DBSCAN clustering on t-SNE features and plot the results with convex hulls.
    
    Params:
        features_tsne (np.ndarray): A numpy array with image file names and their corresponding 2D coordinates
        epsilon (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point
        image_dir (Path): The directory containing the images

    Returns:
        dict: dictionary where keys are cluster labels and values are cluster centers
    """
    
    coordinates_tsne = features_tsne[:, 1:].astype(float)
    
    # create DBScan fit
    db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(coordinates_tsne)

    # labels is an array with the same length has coordinates_tsne with the cluster #
    labels = db.labels_
 
    # append labels as a new column to features_tsne
    features_tsne_with_labels = np.column_stack((features_tsne, labels))
   
    # noise will be clusters that have -1 values
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print("Estimated number of clusters: %d" % n_clusters)
    print("Estimated number of noise points: %d" % n_noise)


    plt.scatter(coordinates_tsne[:, 0], coordinates_tsne[:, 1], c=labels, cmap='viridis')
    cluster_centers = {}
   
    
    clusters_dir = f"{image_dir.replace('extracted_frames', 'clusters')}"
    
    shutil.rmtree(clusters_dir, ignore_errors=True)
    os.makedirs(clusters_dir, exist_ok=True)
    
    for cluster_label in set(labels):
        if cluster_label != -1:  # ignoring nose points
           
            # select images that belong to a certain cluster label
            selected_rows = features_tsne_with_labels[features_tsne_with_labels[:, -1] == str(cluster_label)]
          

            # extract the coordinates (columns 1 and 2)
            cluster_points = selected_rows[:, 1:3].astype(float)
    

            # get the cluster center by finding average of all points and assign number to the feature space
            cluster_center = np.mean(cluster_points, axis=0)
            cluster_centers[cluster_label] = cluster_center
            plt.text(cluster_center[0], cluster_center[1], str(cluster_label),
                    color='red', fontsize=12, weight='bold')
            
            # all hull indices used for this cluster label
            all_hull_indices = []
            # edges of convex hull
            simplices = []

            # more images being processed -> more cluster points -> more convex hulls
            num_hulls = len(cluster_points) // 25
            # minimum is 3
            num_hulls = max(3, num_hulls)
            convex_hull_points = []

            for _ in range(num_hulls):
                
                # can not make convex hull with less than 3 points
                if len(cluster_points) < 3:
                    break
                hull = ConvexHull(cluster_points)
                hull_indices = hull.vertices
                all_hull_indices.append(hull_indices)

                for simplex in hull.simplices:
                    simplices.append((cluster_points[simplex, 0], cluster_points[simplex, 1]))

                convex_hull_points.extend(cluster_points[hull_indices].tolist())
                cluster_points = np.delete(cluster_points, hull_indices, axis=0)

            # plot convex hulls
            for x, y in simplices:
                plt.plot(x, y, 'k-', lw=1)

            # create cluster_label dir
            centroid_dir = os.path.join(clusters_dir, f"cluster_{cluster_label}/")
            os.makedirs(centroid_dir, exist_ok=True)


            for point in convex_hull_points:
                for row in features_tsne_with_labels:
                    if float(row[1]) == point[0] and float(row[2]) == point[1] and row[3] == cluster_label:

                        # move the images with the same coordinates into the new dir
                        shutil.copy(row[0], centroid_dir)
                       
            
                    

    plt.title('DBSCAN Clustering after t-SNE with Convex Hulls')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plot_save_path = os.path.join(f"{image_dir.replace('extracted_frames', '')}", "cluster_plot.jpg")
    plt.savefig(plot_save_path, format='jpg', dpi=300)
    plt.show()
  
    
    return cluster_centers


def export_features(features_dict, filename):
    """
    Export features to a .npy file.
    
    Params:
        features_dict (dict): dictionary where keys are image file names and values are feature arrays
        filename (str): name of the file to save the features
    """
    np.save(filename, features_dict)


def initialize_clustering(image_dir, model_path, frame_skip, is_transformer_model):


  
    image_paths = []

    for image_path in os.listdir(image_dir):
      
        if int((image_path.split('_')[-1]).split('.')[0]) % frame_skip == 0:
            

            image_paths.append(os.path.join(image_dir, image_path))

    # layer 22 is the last layer before the YOLO detection head, choosing highest layer for highest level features for capturing complex patterns
    # layer 27 is last layer in the encoder before the decoder in the RT-Detr Transformer model
    LAYER_INDEX = 27 if is_transformer_model else 22 

    # t-SNE params
    PERPLEXITY = float(ceil(25 * (len(image_paths))/1000))
   
    # DBScan params
    EPSILON = 3 # distance measure to locate points in the neighborhood of any point 
    MIN_SAMPLES = 5 # number of points required in a neighborhood for a point to be considered as a core point


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # ultralytics transformer model 
    model = RTDETR(model_path) if is_transformer_model else YOLO(model_path)
    model.to(device)

    print("Grabbing image features.....")

    SAMPLE_PERCANTAGE = 1
    features = create_features_list(model, image_paths, LAYER_INDEX, SAMPLE_PERCANTAGE)


    print("number of images:", int(len(image_paths) * SAMPLE_PERCANTAGE))
    print("Applying t-SNE.....")
    features_tsne = applying_tsne(features, PERPLEXITY)

    print("Plotting transformed feature space.....")
    plot_reduced_feature_space(features_tsne)

    cluster_centers = cluster_and_plot(features_tsne, EPSILON, MIN_SAMPLES, image_dir)
    
    export_features(features, f"{image_dir.replace('extracted_frames', '')}/features_list.npy")
    
if __name__ == "__main__":
    #image_dir = Path("used_videos\\f_2024_03_14_13_12_21_03\extracted_frames")
    #model_path = "..\best.pt"
    image_dir = ""
    model_path = ""
    initialize_clustering(image_dir, model_path, False) # true if using transformer model (RT DETR)
import json
import math

def get_id(annotation_files, data_type, video_extraction_dir):
    """
    annotation_files (list): a list of the annotation files
    data_type (str): "annotations" or "images", whether to look at the image or annotation id.

    Returns:
    id (int): the unique id of image or annotation
    """

    id_set = set()
    for annotation_file in annotation_files:
        with open(video_extraction_dir + "\\" + annotation_file, 'r') as f:
            data_file = json.load(f)
        if len(data_file["images"]) == 0:
            id = 0
        id_set.update(data["id"] for data in data_file[data_type])
    id = 0
    while id in id_set:
        id += 1
    return id

def cleaning(annotation_files, video_extraction_dir):
    for annotation_file in annotation_files:
        with open(video_extraction_dir + "\\" + annotation_file, 'r') as f:
            data = json.load(f)

        if len(data["images"]) == 0:
            return

        annotated_image_ids = {annotation["image_id"] for annotation in data["annotations"]}

        


        data["images"] = [image_data for image_data in data["images"] if image_data["id"] in annotated_image_ids]
        if annotation_file == "pose_annotations.json":
            data["annotations"] = [annotation_data for annotation_data in data["annotations"] if annotation_data["keypoints"]]
        
        with open(video_extraction_dir + "\\" + annotation_file, 'w') as f:
            json.dump(data, f, indent=4)
        
    return annotated_image_ids

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
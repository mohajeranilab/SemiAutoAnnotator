import json

class AnnotationManager():
    """
    Manages annotation data from the .json files 
    """

    def __init__(self, video_dir, annotation_files):
        """
        Initialize with video directory and a list of annotation files.
        
        Params:
            video_dir (str): the directory containing the video and annotation files
            annotation_files (list): a list of annotation file names
        """

        self.video_dir = video_dir
        self.annotation_files = annotation_files 
        self.id = None

    def cleaning(self): 
        """
        Cleans the annotation files by removing any image entries that do not have corresponding annotations

        Returns:
            annotated_image_ids (set): set of image ids that have annotations
        """

        annotated_image_ids = None
        for annotation_file in self.annotation_files:
            with open(self.video_dir + "\\" + annotation_file, 'r') as f:
                data = json.load(f)

            # if no images exist in the file, skip to the next file
            if len(data["images"]) == 0:
                continue

            # create a set of image ids with annotations
            annotated_image_ids = {annotation["image_id"] for annotation in data["annotations"]}
            
            # filter image data that do not have annotation
            data["images"] = [image_data for image_data in data["images"] if image_data["id"] in annotated_image_ids]

            # for pose, only annotations with keypoints are kept
            if annotation_file == "pose_annotations.json":
                data["annotations"] = [annotation_data for annotation_data in data["annotations"] if annotation_data["keypoints"]]
            
            with open(self.video_dir + "\\" + annotation_file, 'w') as f:
                json.dump(data, f, indent=4)
       
        return annotated_image_ids

    def save_to_json(self, annotations, type):
        """
        Saves annotations made by the user, bbox or pose, to their respective json file

        Params:
            annotations (dict): created annotation to be saved into the respective json file
            type (str): the type of annotation, "bbox" or "pose"

        """ 

        file_index = {
            "bbox": 0,
            "pose": 1
        }.get(type, None)

            
        with open(self.video_dir +"\\" + self.annotation_files[file_index], 'r') as f:
            data = json.load(f)

        image_id = annotations["images"]["id"]

          
        # Check if the image is already present in the images list
        if not any(image["id"] == image_id for image in data["images"]):
            data["images"].append(annotations["images"])


        data["annotations"].append(annotations["annotation"])

        with open(self.video_dir + "\\" + self.annotation_files[file_index], 'w') as f:
            json.dump(data, f, indent=4)
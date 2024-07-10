import json

class AnnotationManager():

    def __init__(self, video_dir, annotation_files):
        self.video_dir = video_dir
        self.annotation_files = annotation_files 
        self.id = None

    def cleaning(self): 
        annotated_image_ids = None
        for annotation_file in self.annotation_files:
            with open(self.video_dir + "\\" + annotation_file, 'r') as f:
                data = json.load(f)

            if len(data["images"]) == 0:
                continue

            annotated_image_ids = {annotation["image_id"] for annotation in data["annotations"]}
            
            
            data["images"] = [image_data for image_data in data["images"] if image_data["id"] in annotated_image_ids]
            if annotation_file == "pose_annotations.json":
                data["annotations"] = [annotation_data for annotation_data in data["annotations"] if annotation_data["keypoints"]]
            
            with open(self.video_dir + "\\" + annotation_file, 'w') as f:
                json.dump(data, f, indent=4)
       
        return annotated_image_ids

    def save_to_json(self, annotations, type):
    
        """
        Saves annotations made by the user, bbox or pose, to their respective json file

        annotations (dict): Created annotation to be saved into the respective json file
        type (str): The type of annotation, "bbox" or "pose"

        """ 

        file_index = {
            "bbox": 0,
            "pose": 1
        }.get(type, None)

            
        with open(self.video_dir +"\\" + self.annotation_files[file_index], 'r') as f:
            data = json.load(f)

        image_id = annotations["images"]["id"]
        if any(image["id"] == image_id for image in data["images"]):
            pass
        else:
            data["images"].append(annotations["images"])

        data["annotations"].append(annotations["annotation"])
        with open(self.video_dir + "\\" + self.annotation_files[file_index], 'w') as f:
            json.dump(data, f, indent=4)
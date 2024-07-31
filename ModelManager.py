import cv2
import os
from pathlib import Path
import shutil
import json
import random
from datetime import datetime
import warnings 
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox

# from PyQtWindows import *
from annotation_labeler import *


class ModelManager():
    """
    The ModelManager class handles the training and prediction processes for an annotation labeler using a YOLO model.
    """
    
    def __init__(self):
        
        self.conf_threshold = 0.25 
        self.correct_dir = "correctly_detected/"
        self.incorrect_dir = "incorrectly_detected/"
         
        

    def predict_all(self, image_dir):
        images = os.listdir(image_dir)
        # Define paths for correct and incorrect directories
        correct_dir_path = os.path.join(self.video_manager.video_dir, self.correct_dir)
        incorrect_dir_path = os.path.join(self.video_manager.video_dir, self.incorrect_dir)


        if os.path.exists(correct_dir_path):
            shutil.rmtree(correct_dir_path)
        if os.path.exists(incorrect_dir_path):
            shutil.rmtree(incorrect_dir_path)

        os.makedirs(correct_dir_path)
        os.makedirs(incorrect_dir_path)

        for image in images:

            bbox_values = self.model.predict(image_dir + "\\" + image, conf=self.conf_threshold)[0].boxes
            
            num_of_objects = len(bbox_values.conf)
            
            conf_list = []
            for i in range(num_of_objects):
                conf = bbox_values.conf[i].item()
                conf_list.append(conf)
                pred_x1, pred_y1, pred_x2, pred_y2 = map(int, bbox_values.xyxy[i].tolist())

            
            conf_list.sort()
            two_highest_mean_conf = conf_list[len(conf_list)-2:]

            if num_of_objects < 1:
                print("no objects detected")
                shutil.copy(image_dir + "\\" + image, incorrect_dir_path)
            elif (pred_y2 - pred_y1) * (pred_x2 - pred_x1) > 5625:
                print("bbox too small, less than 5625 area")
                shutil.copy(image_dir + "\\" + image, incorrect_dir_path)
            elif num_of_objects > 5:
                print("no more than 5 mouses should be in a cage ")
                shutil.copy(image_dir + "\\" + image, incorrect_dir_path)
            elif (sum(two_highest_mean_conf)/2) < 0.5:
                print("Confidences are too low")
                shutil.copy(image_dir + "\\" + image, incorrect_dir_path)
            else:
                shutil.copy(image_dir + "\\" + image, correct_dir_path)


    def retrain(self):
        """
        Will retrain the model on the annotated images of chosen video files
        """

        cv2.destroyAllWindows()
        base_path = Path("labeler_dataset")

        # Create a directory tree in the YOLO format to store the images and labels
        if base_path.exists():
            shutil.rmtree(base_path)
        img_train_path = base_path / "images/train"
        img_val_path = base_path / "images/val"
        label_train_path = base_path / "labels/train"
        label_val_path = base_path / "labels/val"

        img_train_path.mkdir(parents=True, exist_ok=True)
        img_val_path.mkdir(parents=True, exist_ok=True)
        label_train_path.mkdir(parents=True, exist_ok=True)
        label_val_path.mkdir(parents=True, exist_ok=True)

        

        # Select all the video files to train, it will look inside the video folders (if they exist) and retrieve the images and their annotations
        video_paths = []
        print("""\nSelect video files you want to use to train. These video files must have been annotated previously and annotations saved in their directory.\n
    Press Cancel or exit out of the file explorer when finished choosing videos.
        """)
        while True:

            video_path = QFileDialog.getExistingDirectory(
                None,
                "Select Video Folders to Train",
                "used_videos/"
            )

            if video_path == "":
                break
    
            if video_path not in video_paths:
                video_paths.append(video_path)


        if "" in video_paths:
            video_paths.remove("")
        assert len(video_paths) > 0, "There are no video files selected"
    


        for video_path in video_paths:
            assert os.path.exists(os.path.join(video_path, "extracted_frames")), \
                f"No extracted frames for the video folder you have chosen"
            
            if not os.path.exists(os.path.join(video_path, "bbox_annotations.json")):
                warnings.warn(f"There are no bbox annotations made for {video_path}")

            if not os.path.exists(os.path.join(video_path, "pose_annotations.json")):
                warnings.warn(f"There are no pose annotations made for {video_path}")
           
            #video_path + "/bbox"
            with open(os.path.join(video_path, "bbox_annotations.json"), 'r') as f:
                data = json.load(f)

            for image_data in data["images"]:
                shutil.copy(image_data["file_name"], img_train_path)

            for annotation in data["annotations"]:
                image_id = annotation["image_id"]
                for image_data in data["images"]:
                    if image_data["id"] == image_id:
                        file_path = image_data["file_name"]
                        filename = Path(file_path).stem
                        with open(str(label_train_path / filename) + ".txt", 'a') as f:
                            width = (annotation["bbox"][2] - annotation["bbox"][0])/self.img_width

                            height = (annotation["bbox"][3] - annotation["bbox"][1])/self.img_height
                            x_center = (annotation["bbox"][0]/self.img_width) + width/2
                            y_center = (annotation["bbox"][1]/self.img_height) + height/2

                            f.write(f"0 {x_center} {y_center} {width} {height} \n")
                
            

        # Transferring 10% of the train data to val data
        train_images = list(img_train_path.glob("*.jpg"))
        val_count = int(len(train_images) * 0.1)
        val_images = random.sample(train_images, val_count)

        for img in val_images:
            shutil.move(img, img_val_path / img.name)
            label_file = label_train_path / (img.stem + ".txt")

            if label_file.exists():
                shutil.move(label_file, label_val_path / label_file.name)

    
        
        all_entries = img_train_path.iterdir()
        files = [entry for entry in all_entries if entry.is_file()]
        num_of_files = len(files)

        # start model training with specific parameters
        self.model.train(data="annotation_labeler.yaml", epochs = 100, patience=15, degrees=30, shear = 30)
        print("Model has finished training, use the new model weights and run the program again.")
   
    def predicting(self):
        """
        Predict bounding boxes for the given image and save annotations
        """
        
        bbox_values = self.model.predict(self.img_path, conf=self.conf_threshold)[0].boxes
        num_of_objects = len(bbox_values.conf)
        conf_list = []
        for i in range(num_of_objects):
            conf = bbox_values.conf[i].item()
            conf_list.append(conf)
            pred_x1, pred_y1, pred_x2, pred_y2 = map(int, bbox_values.xyxy[i].tolist())

            self.annotation_manager.id = AnnotationTool.get_id(self.annotation_files, self.video_manager, "annotations")
            
            info = {
                "images": {
                    "id": self.img_id,
                    "file_name": self.img_path,
                    "image_height": self.img_height,
                    "image_width": self.img_width
                },
                "annotation": {
                    "id": self.annotation_manager.id,
                    "bbox": [pred_x1, pred_y1, pred_x2, pred_y2],
                    "image_id":self.img_id,
                    "object_id":0,
                    "iscrowd": 0,
                    "area": (pred_y2 - pred_y1) * (pred_x2 - pred_x1),
                    "type": "detected bounding_box",
                    "is_hidden": None, # model won't be able to detect if an image is "hidden"
                    "conf": conf,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            self.annotation_manager.save_to_json(info, "bbox")
            self.annotation_manager.id += 1


        """
        the below portion is code that moves images into either a correct or incorrect directory ONLY when the annotation labeler gets to the image
        """
        # Check and create directories if they do not exist
        # correct_dir_path = os.path.join(self.video_manager.video_dir, self.correct_dir)
        # incorrect_dir_path = os.path.join(self.video_manager.video_dir, self.incorrect_dir)
        # print(correct_dir_path)
        # print(incorrect_dir_path)

        # if not os.path.exists(correct_dir_path):
        #     os.makedirs(correct_dir_path)

        # if not os.path.exists(incorrect_dir_path):
        #     os.makedirs(incorrect_dir_path)
        # conf_list.sort()
        # two_highest_mean_conf = conf_list[len(conf_list)-2:]
        # print(sum(two_highest_mean_conf)/2)

        # if (pred_y2 - pred_y1) * (pred_x2 - pred_x1) > 5625:

        #     print("bbox too small, less than 5625 area")
        #     shutil.copy(self.img_path, incorrect_dir_path)
        # elif num_of_objects > 5:
        #     print("no more than 5 mouses should be in a cage ")
        #     shutil.copy(self.img_path, incorrect_dir_path)

        # elif (sum(two_highest_mean_conf)/2) < 0.5:
        #     print("Confidences are too low")
        #     shutil.copy(self.img_path, incorrect_dir_path)
            
        # else:
        #     shutil.copy(self.img_path, correct_dir_path)
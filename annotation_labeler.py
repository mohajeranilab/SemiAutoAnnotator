import cv2
import os
from pathlib import Path
from tkinter import filedialog
import shutil
import json
import random
import torch
from datetime import datetime
from ultralytics import YOLO
import argparse
import sys
import pygetwindow as gw
import screeninfo
from tkinter import Tk, filedialog, messagebox
import warnings 
from tqdm import tqdm

from clustering import *

class CV2Image():
    def __init__(self, path, name):
        if path:
            self.image = cv2.imread(path)
        if self.image is None:
            raise ValueError("Image could not be loaded.")
        self.path = path
        self.height, self.width, _ = self.image.shape
     
        self.name = name

    def get_image(self):
        """
        Returns the image.
        
        :return: The image array
        """
        #self.image = cv2.imread(self.path)
        return self.image
    
    def set_image(self):
        """
        Sets the image.
        
        :param image: The image array
        """
        self.image = cv2.imread(self.path)
        
class VideoManager():

    def __init__(self, frame_skip, annotation_colors, annotation_files):
        self.video_dir = None
        self.frame_skip = frame_skip
        self.cv2_img = None
        self.annotation_colors = annotation_colors 
        self.model_path = None
        self.annotation_files = annotation_files 

    def extract_frames(self):
        root = Tk()
        root.attributes("-topmost", True)
        root.withdraw()

        video_path = filedialog.askopenfilename(
            initialdir="/", 
            title="SELECT VIDEO FILE",
            filetypes=(("Video files", "*.mp4;*.avi;*.mov;*.mkv"), ("All files", "*.*"))
        )

        if not video_path:
            print("No video file selected.")
            sys.exit()
        
        video_name, _ = os.path.splitext(os.path.basename(video_path))
        self.video_dir = os.path.join("used_videos", video_name)
        
        os.makedirs(self.video_dir, exist_ok=True)

      
        shutil.copy(video_path, "used_videos/" + video_name.split(".")[0])
    
 
        if not os.path.exists(self.video_dir + "/extracted_frames/"):
            os.makedirs(self.video_dir + "/extracted_frames/")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            sys.exit()
            
        frame_count = 0
        extracted_count = 0

        base_name = os.path.splitext(os.path.basename(video_path))[0]
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            frame_filename = os.path.join(self.video_dir + "/extracted_frames/", f"{base_name}_img_{frame_count:05d}.jpg")

            if not os.path.exists(frame_filename):
                ret, frame = cap.read()
                
                if not ret:
                    break
            
            # Skipping by FRAME_SKIP to shorten how many frames are needed to annotate
                if frame_count % self.frame_skip == 0:
                    
                
                    cv2.imwrite(frame_filename, frame) 
                    extracted_count += 1
            
                
            frame_count += self.frame_skip 

        cap.release()
        print(f"Extracted {extracted_count} new frames to 'extracted_frames'")


        self.model_path = filedialog.askopenfilename(initialdir="/", title="SELECT MODEL/WEIGHTS FILE")

        root.destroy()
      
        return video_name
    
    def make_video(self):

        video_path = filedialog.askdirectory(
            initialdir="/", 
            title="SELECT VIDEO FOLDER IN used_videos/",
            
        )

        print("Combining annotated frames to video ......")

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter("output_video.avi", fourcc, 30.0, (self.cv2_img.width, self.cv2_img.height))
        frames_to_write = {}

        # First, gather all frames to write
        for annotation_file in self.annotation_files:
        
            with open(video_path + "\\" + annotation_file, 'r') as f:
                data = json.load(f)
            if len(data["annotations"]) != 0:
                for image_data in data["images"]:
                    frames_to_write[image_data["id"]] = image_data["file_name"]
            else:
                continue
        if not frames_to_write:
            print("No annotations have been made for the selected video")


        all_annotations = {}

        for annotation_file in self.annotation_files:
            with open(video_path + "\\" + annotation_file, 'r') as f:
                data = json.load(f)
            
            for annotation in data["annotations"]:
                image_id = annotation["image_id"]
                if image_id not in all_annotations:
                    all_annotations[image_id] = []
                all_annotations[image_id].append((annotation_file, annotation))

        # Draw all annotations on each frame
        for image_id, image_file in frames_to_write.items():
            if image_id in all_annotations:
                self.cv2_img.set_image()
                
                for annotation_file, annotation in all_annotations[image_id]:
                    if annotation_file == "bbox_annotations.json":
                        cv2.rectangle(self.cv2_img.get_image(), (annotation["bbox"][0], annotation["bbox"][1]), 
                                            (annotation["bbox"][2], annotation["bbox"][3]), 
                                            self.annotation_colors[annotation["object_id"]], 2)

                    elif annotation_file == "pose_annotations.json":
                        for keypoint_annotation in annotation["keypoints"]:
                            if len(keypoint_annotation) != 0:
                                cv2.circle(self.cv2_img.get_image(), (keypoint_annotation[1][0], keypoint_annotation[1][1]), 
                                                5, self.annotation_colors[annotation["object_id"]], -1)

                video.write(self.cv2_img.get_image())

        video.release()

        print("Video has been created called output_video")

class VideoAnnotationTool():
    def __init__(self):
        self.image_dir_path = None
        self.annotation_files = ["bbox_annotations.json", "pose_annotations.json"]
        self.frame_skip = 50
        self.cv2_img = None
        self.img_num = 0
        self.textSizeHeight = None
        self.object_id = 1
        self.already_passed = True
        self.click_count = 0
        self.is_hidden = False
        self.annotations_exists = None
        self.conf_threshold = None
        self.text_to_write = None

        self.annotation_manager = None
        self.video_manager = None
        self.model_manager = None

        self.image_dir = None
        self.model_detecting = "On"
        self.annotation_colors = []
        self.img_id = None

        self.model = None
        self.font_scale = 0.5
        self.font_thickness = 1
        self.font_color = (255, 255, 0)
        self.window_info = {"img_name": None, "coordinates": None, "dimensions": None}

        self.screen_center_x = None
        self.screen_center_y = None
        self.bbox_mode = False
        self.pose_mode = False
        self.bbox_type = None
        self.pose_type = None
        # Define other necessary attributes

   
        # 

    
    
    def dummy_function(self, event, x, y, flags, param):
        pass

   

    def show_image(self): 
        """
        Shows the image, also resizes it to a specific size and also moves it to a specific place on the screen

        ***Moving to a specific place solved my issue of the cv2 window opening in random parts of the screen 
        """
        
        if not any(value is None for value in self.window_info.values()):
            
            if gw.getWindowsWithTitle(self.cv2_img.name):
                
                window = gw.getWindowsWithTitle(self.cv2_img.name)[0]  # Get the first window with the given title
                x, y = window.left, window.top
                
                window_width, window_height = window.width, window.height
                window_width -= 16
                window_height -= 39
    
                self.window_info = {}
                self.window_info["img_name"] = self.cv2_img.name
                self.window_info["coordinates"] = (x, y)
                self.window_info["dimensions"] = (window_width, window_height)

            else:
                
                x, y = self.window_info['coordinates']
                window_width, window_height = self.window_info['dimensions']
        else:
            if gw.getWindowsWithTitle(self.cv2_img.name):
            
                window = gw.getWindowsWithTitle(self.cv2_img.name)[0]  # Get the first window with the given title
                x, y = window.left, window.top
            
                window_width, window_height = window.width, window.height
        
            else:
                x, y = screen_center_x, screen_center_y
    
                window_width, window_height = self.cv2_img.width, self.cv2_img.height

        flag = False
        #cv2.destroyAllWindows()
        if self.window_info["img_name"] != self.cv2_img.name:
            
            self.window_info = {}
            self.window_info["img_name"] = self.cv2_img.name
            self.window_info["coordinates"] = (x, y)
            self.window_info["dimensions"] = (window_width, window_height)
            flag = True
        cv2.namedWindow(self.cv2_img.name, cv2.WINDOW_NORMAL)  
        if flag:
            cv2.resizeWindow(self.cv2_img.name, (1, 1))
            cv2.moveWindow(self.cv2_img.name, -5000, -5000)
        cv2.resizeWindow(self.cv2_img.name, (window_width, window_height))  
    
        cv2.moveWindow(self.cv2_img.name, x, y)


        if self.text_to_write:
            cv2.putText(self.cv2_img.get_image(), self.text_to_write, (int(self.cv2_img.width * 0.05), self.cv2_img.height - int(self.cv2_img.height * 0.05) - self.textSizeHeight), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_color, self.font_thickness)

        cv2.putText(self.cv2_img.get_image(), f"Model: {self.model_detecting}", (int(self.cv2_img.width * 0.75), self.cv2_img.height - int(self.cv2_img.height * 0.05) - self.textSizeHeight), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_color, self.font_thickness)
        
        cv2.imshow(self.cv2_img.name, self.cv2_img.get_image())

    def handle_prev_img(self):

        self.bbox_mode = False
        self.pose_mode = False
        self.already_passed = True
        self.click_count = 0

        if self.img_num > 0:
            self.img_num -= 2 
            cv2.destroyAllWindows()

        elif self.img_num == 0:
            self.img_num -= 1
            cv2.destroyAllWindows()

    @staticmethod
    def get_id(annotation_files, video_manager, data_type):    
        id_set = set()
        for annotation_file in annotation_files:
            with open(video_manager.video_dir + "\\" + annotation_file, 'r') as f:
                data_file = json.load(f)
            id_set.update(data["id"] for data in data_file[data_type])
        id = 0
        while id in id_set:
            id += 1
        return id

    

    def drawing_annotations(self):
        annotation_types = ["bbox", "pose"]

        for type_idx, annotation_type in enumerate(annotation_types):
            with open(self.video_manager.video_dir + "\\" + self.annotation_files[type_idx], 'r') as f:
                annotations = json.load(f)
            
            for annotation in annotations["annotations"]:
                if annotation["image_id"] == self.img_id:
                    if annotation_type == "bbox":
                        cv2.rectangle(self.cv2_img.get_image(), (annotation["bbox"][0], annotation["bbox"][1]), (annotation["bbox"][2], annotation["bbox"][3]), self.annotation_colors[annotation["object_id"]], 2)
                        cv2.putText(self.cv2_img.get_image(), str(annotation["object_id"]), (annotation["bbox"][2] - 10, annotation["bbox"][3]), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_color, self.font_thickness)
                        if annotation["type"] == "detected bounding_box":
                            cv2.putText(self.cv2_img.get_image(), f"{annotation['conf']:.2f}", (annotation["bbox"][0], annotation["bbox"][3]), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_color, self.font_thickness)
                    elif annotation_type == "pose":
                        for keypoint_annotation in annotation["keypoints"]: 
                    
                            if keypoint_annotation[1][0] != None or keypoint_annotation[1][1] != None:
                                cv2.circle(self.cv2_img.get_image(), (keypoint_annotation[1][0], keypoint_annotation[1][1]), 5, self.annotation_colors[annotation["object_id"]], -1)
                                cv2.putText(self.cv2_img.get_image(), keypoint_annotation[0].capitalize(), (keypoint_annotation[1][0], keypoint_annotation[1][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_color, self.font_thickness)


    def drawing_bbox(self, event, x, y, flags, param):
        global start_x, start_y, end_x, end_y
        self.annotation_manager.id = self.get_id(self.annotation_files, self.video_manager, "annotations")
        self.img_id = None
        for annotation_file in self.annotation_files:
            with open(self.video_manager.video_dir + "\\" + annotation_file, 'r') as f:
                data = json.load(f)

              
                break_loop = False
                for image_data in data["images"]:
                    if image_data["file_name"] == self.cv2_img.path:
               
                        self.img_id = image_data["id"]
                        break_loop = True
                        break
                if break_loop:
                    break
        if self.img_id == None:
            self.img_id = self.get_id(self.annotation_files, self.video_manager, "images")
    

        if self.click_count == 1:
            self.cv2_img.set_image()
            self.drawing_annotations()

            cv2.rectangle(self.cv2_img.get_image(), (start_x, start_y), (x, y), self.annotation_colors[self.object_id], 2)

            if self.is_hidden == 1:
        
                self.text_to_write = f"Bounding Box Mode - Hidden - {self.object_id}"
            elif self.bbox_type == "feces":
                self.text_to_write = f"Bounding Box Mode - Feces"
            elif self.bbox_type == "normal":
                self.text_to_write = f"Bounding Box Mode - {self.object_id}"
            if start_x > x:
                x = start_x
            if start_y > y:
                y = start_y
            cv2.putText(self.cv2_img.get_image(), str(self.object_id), (x - 10, y), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_color, self.font_thickness)
            self.show_image()
            
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.click_count == 0:
            
                start_x = max(0, min(x, self.cv2_img.width))  
                start_y = max(0, min(y, self.cv2_img.height))   
                self.click_count += 1

        elif event == cv2.EVENT_LBUTTONUP:
        
            end_x, end_y = x, y
            self.click_count = 0
                  
            end_x = max(0, min(end_x, self.cv2_img.width))
            end_y = max(0, min(end_y, self.cv2_img.height))
            if end_x < start_x:
                start_x, end_x = end_x, start_x

            if end_y < start_y:
                start_y, end_y = end_y, start_y

            info = {
                "images": {
                "id": self.img_id,
                "file_name": self.cv2_img.path,
                "image_height": self.cv2_img.height,
                "image_width": self.cv2_img.width
                },
                "annotation": {
                    "id": self.annotation_manager.id,
                    "bbox": [start_x, start_y, end_x, end_y],
                    "image_id":self.img_id,
                    "object_id":self.object_id,
                    "iscrowd": 0,
                    "area": (end_x - start_x) * (end_y - start_y),
                    "type": self.bbox_type + " " + "bounding_box",
                    "is_hidden": self.is_hidden,
                    "conf": 1,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                }
            }
           
       
            cv2.rectangle(self.cv2_img.get_image(), (start_x, start_y), (end_x, end_y), self.annotation_colors[self.object_id], 2)
            
            self.show_image()
            self.annotation_manager.save_to_json(info, "bbox")


    def drawing_pose(self, event, x, y, flags, param):

        with open(self.video_manager.video_dir + "/pose_annotations.json", 'r') as f:
            data = json.load(f)

        if event == cv2.EVENT_LBUTTONDOWN:
            point = (x, y)
            cv2.circle(self.cv2_img.get_image(), (point[0], point[1]), 5, self.annotation_colors[self.object_id], -1)

            cv2.putText(self.cv2_img.get_image(), self.pose_type.capitalize(), (point[0], point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_color, self.font_thickness)
            self.show_image()
            to_append = (self.pose_type, (point))    
            for annotation in data["annotations"]:
                if annotation["id"] == self.annotation_manager.id:
                    annotation["keypoints"].append(to_append)
                    annotation["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    break
                
            with open(self.video_manager.video_dir + "/pose_annotations.json", 'w') as f:
                json.dump(data, f, indent = 4)



    

    def update_img_with_id(self):
        # reread the image but with a new object id and the same bbox titles as before 
        if self.bbox_mode == True:
            self.cv2_img.set_image()
        
            self.drawing_annotations()
            if self.is_hidden == 1:
                self.text_to_write = f"Bounding Box Mode - Hidden - {self.object_id}"
            elif self.bbox_type == "feces":
                self.text_to_write = f"Bounding Box Mode - Feces"
            elif self.bbox_type == "normal":
                self.text_to_write = f"Bounding Box Mode - {self.object_id}"
            

            self.show_image()

        # initialize a new pose annotation when a new object id is created 
        elif self.pose_mode == True:
        
            self.cv2_img.set_image()
            self.drawing_annotations()

            pose_mode_text = f"Pose Mode - {self.object_id}"
            if self.pose_type:
                pose_mode_text = f"Pose Mode - {self.pose_type.capitalize()} - {self.object_id}"
                self.annotation_manager.id = self.get_id(self.annotation_files, self.video_manager, "annotations")
                info = {
                    "images": {
                        "id": self.img_id,
                        "file_name": self.cv2_img.path,
                        "image_height": self.cv2_img.height,
                        "image_width": self.cv2_img.width
                    },
                    "annotation": {
                        "id": self.annotation_manager.id,
                        "keypoints": [],
                        "image_id": self.img_id,
                        "object_id": self.object_id,
                        "iscrowd": 0,
                        "type": "pose",
                        "conf": 1,
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                }
                self.annotation_manager.save_to_json(info, "pose")

            self.text_to_write = pose_mode_text
            self.show_image()

    def annotating(self):

        self.cv2_img.set_image()
        self.bbox_mode = False
        self.pose_mode = False
        self.object_id = 1
        self.is_detected = False
        self.is_hidden = 0
        self.img_id = None
        self.click_count = 0
        
        for annotation_file in self.annotation_files:

            with open(self.video_manager.video_dir + "\\" + annotation_file, 'r') as f:
                data = json.load(f)

                for image_data in data["images"]:
                    if image_data["file_name"] == self.cv2_img.path:
                        self.img_id = image_data["id"]

        if self.img_id == None:
            self.img_id = self.get_id(self.annotation_files, self.video_manager, "images")

        with open(self.video_manager.video_dir + "/bbox_annotations.json", 'r') as f:
            data = json.load(f)

            for annotation in data["annotations"]:
                if annotation["image_id"] == self.img_id:
                    if annotation["type"] == "detected bounding_box":
                        self.is_detected = True
                        break
            

            if self.is_detected == False and self.video_manager.model_path != "" and self.video_manager.model_path != None and not isinstance(self.video_manager.model_path, tuple) and self.model_detecting == "On":
                self.model_manager = ModelManager(self.model, self.cv2_img.path, self.cv2_img, self.conf_threshold, self.cv2_img.width, self.cv2_img.height, self.img_id, self.object_id, self.annotation_manager, self.video_manager, self.annotation_files)
                self.model_manager.predicting()
              
    
        
            self.drawing_annotations()
            self.text_to_write = None 
            self.show_image()

            while True:
                key = cv2.waitKey(1)
        
    
                if key == 27 or cv2.getWindowProperty(self.cv2_img.name, cv2.WND_PROP_VISIBLE) < 1: # "Escape": Exits the program 
                    self.annotation_manager.cleaning()
                

                    sys.exit()
                
                elif key == ord('r'):
                    self.retrain()
            
                    
                elif key == ord('j'): # "J": Previous object ID
                    
                    self.object_id -= 1 if self.object_id > 1 else 0 
                    self.update_img_with_id()
                elif key == ord('v'): # make video
                    
                    self.make_video()

                elif key == ord('m'): # "M": Turns model detection on or off, as shown on the image
                    self.cv2_img.set_image()
                    self.model_detecting = "Off" if self.model_detecting == "On" else "On"
                    self.drawing_annotations()
                    self.show_image()

                elif key == ord('b'): # bbox mode
                    if self.bbox_mode == False: 
                        self.bbox_mode = True
                        self.click_count = 0
                        self.cv2_img.set_image()
                        self.drawing_annotations() 
                        self.text_to_write = f"Bounding Box Mode - {self.object_id}"
                        
                        self.pose_mode = False
                        self.bbox_type = "normal"
                        self.show_image()
                        cv2.setMouseCallback(self.cv2_img.name, self.drawing_bbox)  # Enable mouse callback for keypoint placement
                    

                    else:
                        self.bbox_mode = False
                        self.bbox_type = "normal"
                        self.is_hidden = 0 
                        self.text_to_write = None
                        self.cv2_img.set_image()
                        self.drawing_annotations()
                        self.show_image()
                        cv2.setMouseCallback(self.cv2_img.name, self.dummy_function)
                
                elif key == ord('p'): # pose mode
                    if self.pose_mode == False:
                        self.pose_mode = True
                        self.click_count = 0
                        self.cv2_img.set_image()
                        self.drawing_annotations()
                        self.text_to_write = f"Pose Mode - {self.object_id}"
                        
                        self.show_image()
                    
                        self.bbox_mode = False
                        self.pose_type = ""
                        self.annotation_manager.id = self.get_id(self.annotation_files, self.video_manager, "annotations")
            
                        info = {
                            "images": {
                                "id": self.img_id,
                                "file_name": self.cv2_img.path,
                                "image_height": self.cv2_img.height,
                                "image_width": self.cv2_img.width
                            },
                            "annotation": {
                                "id": self.annotation_manager.id,
                                "keypoints": [],
                                "image_id":self.img_id,
                                "object_id":self.object_id,
                                "iscrowd": 0,
                                "type": "pose",
                                "conf": 1,
                                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                        }
                        self.annotation_manager.save_to_json(info, "pose")
                        cv2.setMouseCallback(self.cv2_img.name, self.dummy_function)
                    else:
                        self.text_to_write = None
                        self.pose_mode = False
                        self.cv2_img.set_image()
                        self.drawing_annotations()
                        self.show_image()
                        cv2.setMouseCallback(self.cv2_img.name, self.dummy_function)


                elif key == 13: # enter; next image in dataset
                    #cv2.destroyAllWindows()
                    self.pose_mode = False
                    self.bbox_mode = False
                    self.already_passed = False
                    self.object_id = 1
                    self.show_image()
                    cv2.destroyAllWindows()
                    return

                elif key == 8: # backspace; prev_img 
                    self.show_image()
                    self.handle_prev_img()
                    return

                elif key == ord('d'): # delete all annotations for an image
                    for annotation_file in self.annotation_files:
                        with open(self.video_manager.video_dir + "\\" + annotation_file, 'r') as f:
                            data = json.load(f)
                        data["annotations"] = [annotation for annotation in data["annotations"] if annotation["image_id"] != self.img_id]
            
                        with open(self.video_manager.video_dir + "\\" + annotation_file, 'w') as f:
                            json.dump(data, f, indent=4)

                    self.cv2_img.set_image()
                    self.text_to_write = None
                    self.show_image()
                    cv2.setMouseCallback(self.cv2_img.name, self.dummy_function)
                    self.bbox_mode = False
                    self.pose_mode = False
                    self.object_id = 1

                elif key == 26: # ctrl + z; undo
                    is_empty = True

                    for annotation_file in self.annotation_files:
                        with open(annotation_file, 'r') as f:
                            data = json.load(f)
            
                        if any(annotation["image_id"] == self.img_id for annotation in data["annotations"]):
                            is_empty = False
                            break
                    
                    if is_empty:
                        self.show_image()
                        self.handle_prev_img()
                        self.object_id = 1
                        return

                    else:
                        latest_time = None
                    
                    for annotation_file in self.annotation_files:
                        with open(annotation_file, 'r') as f:
                            data = json.load(f)
                        
                        for i, annotation in enumerate(data["annotations"]):
                            if annotation["image_id"] == self.img_id:
                                timestamp = datetime.strptime(annotation["time"], "%Y-%m-%d %H:%M:%S")
                                if latest_time is None or timestamp > latest_time:
                                    latest_time = timestamp

            
                    for annotation_file in self.annotation_files:
                        with open(annotation_file, 'r') as f:
                            data = json.load(f)
                        
                        for i in range(len(data["annotations"])):
                            timestamp = datetime.strptime(data["annotations"][i]["time"], "%Y-%m-%d %H:%M:%S")
                            if timestamp == latest_time:
                                self.object_id = data["annotations"][i]["object_id"]
                                if data["annotations"][i]["type"] == "pose":
                                    if len(data["annotations"][i]["keypoints"]) != 0:
                                        data["annotations"][i]["keypoints"].pop()
                                        break
                                    else:
                                        
                                        del data["annotations"][i]
                                        break
                                else:
                
                                    del data["annotations"][i]
                                break
                        
                        with open(annotation_file, 'w') as f:
                            json.dump(data, f, indent=4)
                    self.cv2_img.set_image()
                    self.drawing_annotations()
                    # rewriting the previous titles after deletion
                    mode_text = ""
                    if self.bbox_mode:
                        mode_text = "Bounding Box Mode - "
                        if self.is_hidden:
                            mode_text += "Hidden - "
                        elif self.bbox_type == "feces":
                            mode_text += "Feces - "
                        mode_text += str(self.object_id)
                    elif self.pose_mode:
                        mode_text = "Pose Mode - "
                        if self.pose_type:
                            mode_text += f"{self.pose_type.capitalize()} - "
                        mode_text += str(self.object_id)

                
                    self.already_passed = False
                    self.drawing_annotations()
                    self.text_to_write = mode_text
                    self.show_image()
                    
                elif key == ord('n'): # next mouse ID
                    self.object_id += 1

                    self.update_img_with_id()
                    

                
                if self.bbox_mode:
    
                    bbox_options = {
                        ord('f'): ("feces", "Feces"),
                        ord('h'): ("normal", "Hidden")
                    }

                    for keybind, (bbox_label, mode_message) in bbox_options.items():
                        if key == keybind:
                            self.cv2_img.set_image()
                            self.drawing_annotations()
                            self.text_to_write = f"Bounding Box Mode - {mode_message} - {self.object_id}"
                            
                            self.show_image()
                        
                            self.is_hidden = 1 if bbox_label == "normal" else 0
                            self.bbox_type = bbox_label.lower()
                            cv2.setMouseCallback(self.cv2_img.name, self.drawing_bbox)
                    

                elif self.pose_mode:

                    pose_options = {
                    ord('1'): ("Head"),
                    ord('2'): ("Tail"),
                    ord('3'): ("Neck")
                }

                    for keybind, p_label in pose_options.items():
                        if key == keybind:
                            self.cv2_img.set_image()
                            self.drawing_annotations()
                            self.text_to_write = f"Pose Mode - {p_label} - {self.object_id}"
                            
                            self.show_image()
                            self.pose_type = p_label.lower()
                            cv2.setMouseCallback(self.cv2_img.name, self.drawing_pose)
                


        
        self.img_num += 1
        cv2.destroyAllWindows()

    def run_tool(self):
      

        # initializing constants 
        parser = argparse.ArgumentParser()
        parser.add_argument("--frame_skip", type=int, default=50, help="Number of frames to skip")
        parser.add_argument("--model_path", type=str, default=None, help="Path to the model/weights file ")
        args = parser.parse_args()
        self.frame_skip = args.frame_skip
        self.model_path = args.model_path

        self.annotation_files = ["bbox_annotations.json", "pose_annotations.json"]
        self.font_scale = 0.5
        self.font_thickness = 1
        self.font_color = (255, 255, 0)
        self.annotation_colors = []
        self.conf_threshold = 0.25
        
        screen = screeninfo.get_monitors()[0]  # Assuming you want the primary monitor
        width, height = screen.width, screen.height
        self.screen_center_x = int((width - 700) / 2)
        self.screen_center_y = int((height - 500)/ 2)
        # creating a list of random annotation colors that are- the same throughout different runs 
        seed = 42
        random.seed(seed)
        for _ in range(30):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)

            color = (r, g, b)
            self.annotation_colors.append(color)

        self.window_info = {"img_name": None, "coordinates": None, "dimensions": None}
        # to get the sizing for putting text at appropiate places on the cv2 window
        textSize, baseline = cv2.getTextSize("test", cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_thickness)
        textSizeWidth, self.textSizeHeight = textSize

        self.video_manager = VideoManager(self.frame_skip, self.annotation_colors, self.annotation_files)
        # user will choose the video file and/or model file, video file will be extracted as frames into a image directory
        #video_name = self.extract_frames()
        video_name = self.video_manager.extract_frames()
        self.image_dir = "used_videos/" + video_name.split(".")[0] + "/extracted_frames/"

        assert self.image_dir is not None, "A image folder was empty."

        # initialize the json files in the respective video directory
        for annotation_file in self.annotation_files:
            if not os.path.exists(self.video_manager.video_dir + "\\" + annotation_file):
                json_content = {"images": [], "annotations": []}
                
                with open(self.video_manager.video_dir + "\\" + annotation_file, 'w') as f:
                    json.dump(json_content, f, indent=4)
        dir_list = None
   
        # if a model is selected, otherwise let the user annotate with no model assistance
        if not isinstance(self.video_manager.model_path, tuple) and self.video_manager.model_path != "" and self.video_manager.model_path != None:
            print("CUDA available?: ", torch.cuda.is_available())
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            self.model = YOLO(self.video_manager.model_path)
            self.model.to(device)

            # comment this to turn off clustering 
            # initialize_clustering(Path(self.image_dir), self.video_manager.model_path)
            # dir_list = os.listdir("used_videos/" + video_name.split(".")[0] + "/clusters/")
            # for i, dir in enumerate(dir_list):
            #     dir_list[i] = "used_videos/" + video_name.split(".")[0] + "/clusters/" + dir + "/" 
            # # delete extracted_frames to save space
            # shutil.rmtree(self.image_dir, ignore_errors=True)
        else:
            dir_list = None
            
        self.already_passed = False
        self.object_id = 1
        self.annotations_exists = False
        self.model_detecting = "On"

        for annotation_file in self.annotation_files:
            with open(self.video_manager.video_dir + "\\" + annotation_file, 'r') as f:
                data = json.load(f)

            if data["annotations"]:
                self.annotations_exists = True
                break


        # Create a window pop up to determine if user wants to continue where they left off 
        if self.annotations_exists:
            window = Tk()
            window.attributes('-topmost', True)
            window.withdraw()
            result = messagebox.askquestion("Continue to Next Image", "Do you want to continue your work on the image following the last annotated image?")

            window.destroy()

        self.img_num = 0
    
        # directory list will be the list of clusters if a model is chosen, or a list of extracted frames
        directories = [self.image_dir] if not dir_list else dir_list
        self.annotation_manager = AnnotationManager(self.video_manager.video_dir, self.annotation_files)
        
        for current_dir in directories:
            
            imgs = os.listdir(current_dir)
            

            with tqdm(total=len(imgs), desc=f"cluster {(current_dir.split('_')[-1]).replace('/', '')}") as pbar:
                while self.img_num < len(imgs):
                    
                    self.is_hidden = 0
                    self.annotations_exists = False
                    annotated_image_ids = set()
                    self.img_num = int(self.img_num)
                    imagepath = os.path.join(current_dir, imgs[int(self.img_num)])
                    imagename = os.path.basename(imagepath)
                    self.cv2_img = CV2Image(imagepath, imagename)
            
                    if int(((self.cv2_img.name.split('_'))[-1]).replace('.jpg', '')) % self.frame_skip == 0:
                      
                        self.cv2_img = CV2Image(imagepath, imagename)
                  
                        annotated_image_ids = self.annotation_manager.cleaning()
                    
                        if annotated_image_ids and self.already_passed == False:
                            for annotation_file in self.annotation_files:
                                
                                with open(os.path.join(self.video_manager.video_dir, annotation_file), 'r') as f:
                                    data = json.load(f)

                                if len(data["images"]) == 0:
                                    continue

                                for image_data in data["images"]:
                                    if image_data["file_name"] == self.cv2_img.path:
                                        if image_data["id"] in annotated_image_ids:
                                            self.annotations_exists = True
                                            continue
                     
                        if not self.annotations_exists:
                            self.annotating()
                        else:
                            if result == "no":
                                self.annotating()
                        
                    
                    self.img_num += 1
                    pbar.n = self.img_num
                    pbar.refresh()

            self.img_num = 0  # reset img_num for the next directory    
         
        cv2.destroyAllWindows()

class ModelManager:
    def __init__(self, model, image_path, img, conf_threshold, img_width, img_height, image_id, object_id, annotation_manager, video_manager, annotation_files):
        self.image_path = image_path
        self.conf_threshold = conf_threshold
        self.model = model
        self.img_width = img_width
        self.img_height = img_height
        self.img = img
        self.image_id = image_id 
        self.object_id = object_id
        self.annotation_manager = annotation_manager
        self.video_manager = video_manager
        self.annotation_files = annotation_files

    def retrain(self):
        """
        Will retrain on the annotated images of chosen video files

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


        root = Tk()
        root.attributes('-topmost', True)
        root.withdraw()

        # Select all the video files to train, it will look inside the video folders (if they exist) and retrieve the images and their annotations
        video_paths = []
        print("""\nSelect video files you want to use to train. These video files must have been annotated previously and annotations saved in their directory.\n
    Press Cancel or exit out of the file explorer when finished choosing videos.
        """)
        while True:
            video_path = filedialog.askdirectory(
                initialdir="used_videos/",
                title="SELECT VIDEO FOLDERS TO TRAIN"
            )

            if video_path == "":
                break
    
            if video_path not in video_paths:
                video_paths.append(video_path)

        root.destroy()
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
            # assert os.path.exists(os.path.join(video_path, "bbox_annotations.json")), \
            #     f"There are no annotations made for {video_path}"
            
            with open(video_path + "/bbox_annotations.json", 'r') as f:
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

        
        self.model.train(data="annotation_labeler.yaml", epochs = 100, patience=15, degrees=30, shear = 30)
        print("Model has finished training, use the new model weights and run the program again.")
        sys.exit()
   
    def predicting(self):
        
        bbox_values = self.model.predict(self.image_path, conf=self.conf_threshold)[0].boxes
        num_of_objects = len(bbox_values.conf)
        conf_list = []
        for i in range(num_of_objects):
            conf = bbox_values.conf[i].item()
            conf_list.append(conf)
            pred_x1, pred_y1, pred_x2, pred_y2 = map(int, bbox_values.xyxy[i].tolist())

            self.annotation_manager.id = VideoAnnotationTool.get_id(self.annotation_files, self.video_manager, "annotations")
            
            info = {
                "images": {
                    "id": self.image_id,
                    "file_name": self.image_path,
                    "image_height": self.img_height,
                    "image_width": self.img_width
                },
                "annotation": {
                    "id": self.annotation_manager.id,
                    "bbox": [pred_x1, pred_y1, pred_x2, pred_y2],
                    "image_id":self.image_id,
                    "object_id":self.object_id,
                    "iscrowd": 0,
                    "area": (pred_y2 - pred_y1) * (pred_x2 - pred_x1),
                    "type": "detected bounding_bbox",
                    "conf": conf,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            self.annotation_manager.save_to_json(info, "bbox")
            self.annotation_manager.id += 1

        conf_list.sort()
        two_highest_mean_conf = conf_list[len(conf_list)-2:]
        # print(sum(two_highest_mean_conf)/2)
        # if (pred_y2 - pred_y1) * (pred_x2 - pred_x1) > 5625:

        #     print("bbox too small, less than 5625 area")
        #     shutil.copy(self.image_path, self.video_manager.video_dir[])
        # elif num_of_objects > 5:
        #     print("no more than 5 mouses should be in a cage ")
        #     shutil.copy(self.image_path, self.video_manager.video_dir[])

        # elif (sum(two_highest_mean_conf)/2) < 0.5:
        #     print("Confidences are too low")
        #     shutil.copy(self.image_path, self.video_manager.video_dir[])
            
        # else:
        #     shutil.copy(self.image_path, self.video_manager.video_dir[])



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
    

if __name__ == "__main__":
    screen = screeninfo.get_monitors()[0]  # Assuming you want the primary monitor
    width, height = screen.width, screen.height
    screen_center_x = int((width - 700) / 2)
    screen_center_y = int((height - 500)/ 2)
    tool = VideoAnnotationTool()
    tool.run_tool()
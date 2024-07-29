import cv2
import os
from pathlib import Path
import shutil
import json
import random
from datetime import datetime, timedelta
import argparse
import sys
import screeninfo
from tkinter import Tk, filedialog, messagebox
from tqdm import tqdm
import pywinctl as pwc
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
import copy

from VideoManager import *
from AnnotationManager import *
from ModelManager import * 
from PyQtWindows import ButtonWindow

class CV2Image():
    def __init__(self, path, name):
        """
        Initializes a CV2Image object with the given image path and name.

        Params:
            path (str): The file path to the image
            name (str): The name of the image

        Raises:
             ValueError: If the image cannot be loaded.
        """

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
        
        Returns:
           image (np.array): image read using cv2
        """

        return self.image
    
    def set_image(self):
        """
        Sets the image.
        """

        self.image = cv2.imread(self.path)
        

class AnnotationTool():
    def __init__(self):
    
        self.annotation_files = ["bbox_annotations.json", "pose_annotations.json"]
       
        self.img_num = 0
        self.textSizeHeight = None
        self.object_id = 1
        self.already_passed = True
        self.click_count = 0
        self.is_hidden = False
        self.annotations_exists = None
        self.text_to_write = None


        self.annotation_manager = None
        self.video_manager = None
        self.model_manager = None

        self.image_dir = None

        self.annotation_colors = []
        self.img_id = None
        self.corner_size = 10

        self.font_scale = 0.5
        self.font_thickness = 0
        self.font_color = (255, 255, 0)
        self.window_info = {"img_name": None, "coordinates": None, "dimensions": None}
   
        self.editing_mode = False
        self.bbox_mode = False
        self.pose_mode = False
        self.bbox_type = None
        self.pyqt_button_window = ButtonWindow()
   
        self.pose_type = None
        self.current_dir = None
        self.imgs = None 

        self.redo_stack = []
        self.pyqt_button_window.setWindowFlags(Qt.WindowStaysOnTopHint)

    
    
    def dummy_function(self, event, x, y, flags, param):
        pass


    def show_image(self): 
        """
        Shows the image, also resizes it to a specific size and also moves it to a specific place on the screen

        *** An issue I ran into is that when creating a new opencv2 window, it creates the window somewhere (previous position of the last opened cv2 window ?)
        *** THEN moves the window to the specified location. This creates an undesired flickering effect.
        """
        
        if not any(value is None for value in self.window_info.values()):
            
           
            if pwc.getWindowsWithTitle(self.cv2_img.name):
                window = pwc.getWindowsWithTitle(self.cv2_img.name)[0]
                x, y = window.left, window.top
                
                window_width, window_height = window.width, window.height
                window_width -= 16
                window_height -= 39
                self.window_info = {}
                self.window_info["img_name"] = self.cv2_img.name
                self.window_info["coordinates"] = (x, y)
                self.window_info["dimensions"] = (window_width, window_height)
            else:
                x, y = self.window_info["coordinates"]
                window_width, window_height = self.window_info["dimensions"]
        else:
            if pwc.getWindowsWithTitle(self.cv2_img.name):
                window = pwc.getWindowsWithTitle(self.cv2_img.name)[0]
                x, y = window.left, window.top

                window_width, window_height = window.width, window.height
        
            else:
                x, y = self.screen_center_x, self.screen_center_y
    
                window_width, window_height = self.cv2_img.width, self.cv2_img.height

        flag = False
        
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
        self.pyqt_button_window.move_to_coordinates(x - 200, y)

        if self.text_to_write:
            cv2.putText(self.cv2_img.get_image(), self.text_to_write, (int(self.cv2_img.width * 0.05), self.cv2_img.height - int(self.cv2_img.height * 0.05) - self.textSizeHeight), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_color, self.font_thickness)

        cv2.putText(self.cv2_img.get_image(), f"Model: {self.model_detecting}", (int(self.cv2_img.width * 0.75), self.cv2_img.height - int(self.cv2_img.height * 0.05) - self.textSizeHeight), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_color, self.font_thickness)
        
        cv2.imshow(self.cv2_img.name, self.cv2_img.get_image())
        cv2.setWindowProperty(self.cv2_img.name, cv2.WND_PROP_TOPMOST, 1)
    
        
    @staticmethod
    def move_to(window_name, x, y):
        """
        Move a OpenCV window to a specific position on the screen.

        Params:
            window_name (str): The name of the window to move
            x (int): The x-coordinate of the new position
            y (int): The y-coordinate of the new position
        """
        cv2.moveWindow(window_name, x, y)


    def handle_prev_img(self):
        """
        Handle navigation to the previous image in the sequence
        """
        self.bbox_mode = False
        self.pose_mode = False
        self.editing_mode = False
        self.already_passed = True
        self.click_count = 0


        if self.img_num == 0:
            self.img_num -= 1
            cv2.destroyAllWindows()
            return

        self.img_num -= 1 
        while self.img_num < len(self.imgs):
            img_path = os.path.join(self.current_dir, self.imgs[self.img_num])
            img_name = os.path.basename(img_path)
            if int(((img_name.split('_'))[-1]).replace('.jpg', '')) % self.frame_skip != 0:

                if self.img_num > 0:
                    self.img_num -= 1 
        
            else:
                self.img_num -= 1
                cv2.destroyAllWindows() 
                return


    @staticmethod
    def get_id(annotation_files, video_manager, data_type):    
        """
        Generate a unique id

        Params:
            annotation_files (list): list of annotation files
            video_manager (VideoManager): Instance of VideoManager class to retrieve the video_dir
            data_type (str): type of data to get ids (images or annotations)

        Returns:
            id (int): unique id 
        """

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
        """
        Draw annotations on the current image from the .json files
        """

        annotation_types = ["bbox", "pose"]

        for annotation_file in self.annotation_files:
            with open(self.video_manager.video_dir + "\\" + annotation_file, 'r') as f:
        
                annotations = json.load(f)
            
            for annotation in annotations["annotations"]:
                if annotation["image_id"] == self.img_id:
            
                    if annotation["type"].split()[-1] == "bounding_box":
                   
                        corner_points = [(annotation["bbox"][0], annotation["bbox"][1]), (annotation["bbox"][2], annotation["bbox"][1]), (annotation["bbox"][0], annotation["bbox"][3]), (annotation["bbox"][2], annotation["bbox"][3])]
                        for corner_x, corner_y in corner_points:
                            cv2.rectangle(self.cv2_img.get_image(), (corner_x - self.corner_size//2 , corner_y - self.corner_size//2), (corner_x + self.corner_size//2, corner_y + self.corner_size//2), self.annotation_colors[annotation["object_id"]], 2)
                        cv2.rectangle(self.cv2_img.get_image(), (annotation["bbox"][0], annotation["bbox"][1]), (annotation["bbox"][2], annotation["bbox"][3]), self.annotation_colors[annotation["object_id"]], 2)
                        cv2.putText(self.cv2_img.get_image(), str(annotation["object_id"]), (annotation["bbox"][2] - 20, annotation["bbox"][3] - 5), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_color, self.font_thickness)
                        if annotation["type"] == "detected bounding_box":
                            cv2.putText(self.cv2_img.get_image(), f"{annotation['conf']:.2f}", (annotation["bbox"][0], annotation["bbox"][3]), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_color, self.font_thickness)
               
                    elif annotation["type"] == "pose":
                        for keypoint_annotation in annotation["keypoints"]: 
                    
                            if keypoint_annotation[1][0] != None or keypoint_annotation[1][1] != None:
                                cv2.circle(self.cv2_img.get_image(), (keypoint_annotation[1][0], keypoint_annotation[1][1]), 5, self.annotation_colors[annotation["object_id"]], -1)
                                cv2.putText(self.cv2_img.get_image(), keypoint_annotation[0].capitalize(), (keypoint_annotation[1][0], keypoint_annotation[1][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale-0.25, self.font_color, self.font_thickness)
                        if annotation['keypoints']:
                            keypoints = {kp[0]: kp[1] for kp in annotation['keypoints']}
          
                       
                            keypoint_pairs = [
                                ("head", "neck"),
                                ("neck", "tail"),
                                ("neck", "l hand"),
                                ("neck", "r hand"),
                                ("tail", "l leg"),
                                ("tail", "r leg")
                            ]

                            for key1, key2 in keypoint_pairs:
                                if key1 in keypoints and key2 in keypoints:
                                    pt1 = tuple(keypoints[key1])
                                    pt2 = tuple(keypoints[key2])
                                    cv2.line(self.cv2_img.get_image(), pt1, pt2, self.annotation_colors[annotation["object_id"]], thickness=1)


    def editing(self, event, x, y, flags, param):
        """
        Handle editing of annotations based on mouse events

        Args:
            event (int): type of mouse event (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP).
            x (int): x-coordinate of the mouse event
            y (int): y-coordinate of the mouse event
            flags (int): flags passed by OpenCV
            param (any): additional parameters (not used)

        Global Variables:
            move_top_left (bool): indicates if the top-left corner of a bounding box is being moved
            move_top_right (bool): indicates if the top-right corner of a bounding box is being moved
            move_bottom_right (bool): indicates if the bottom-right corner of a bounding box is being moved
            move_bottom_left (bool): indicates if the bottom-left corner of a bounding box is being moved
            move_pose_point (bool): indicates if a pose keypoint is being moved
            file_to_dump (str): file to update with new annotation data
            moved (bool): indicates if an annotation has been moved
        """

        global move_top_left, move_top_right, move_bottom_right, move_bottom_left, move_pose_point, file_to_dump, moved
        if event == cv2.EVENT_LBUTTONDOWN:
            self.temp_bbox_coords = None
            breakout = False
            if self.click_count == 0:
                for annotation_file in self.annotation_files:
                    if breakout:
                        break
                    with open(self.video_manager.video_dir + "\\" + annotation_file, 'r') as f:
                        data = json.load(f)

                    img_id = next((img_data["id"] for img_data in data["images"] if img_data["file_name"] == self.cv2_img.path), None)
                    if img_id is None:
                        continue

                    for annotation_data in data["annotations"]:
                        if annotation_data["image_id"] != img_id:
                            continue

                        if annotation_data["type"] == "pose":
                            if len(annotation_data["keypoints"]) == 0:
                                continue

                            self.temp_keypoints = annotation_data["keypoints"]
                            for keypoint in self.temp_keypoints:
                                if abs(keypoint[1][0] - x) < 7.5 and abs(keypoint[1][1] - y) < 7.5:
                                    self.keypoint_type = keypoint[0]
                                    self.keypoint_value = keypoint[1]
                                    self.annotation_manager.id = annotation_data["id"]
                                    move_pose_point = True
                                    move_top_left = move_top_right = move_bottom_left = move_bottom_right = False
                                    breakout = True
                                    break
                            else:
                                self.keypoint_type = self.keypoint_value = None
                                move_pose_point = False
                        else:
                            corners = {
                                "top_left": (annotation_data["bbox"][0], annotation_data["bbox"][1]),
                                "top_right": (annotation_data["bbox"][2], annotation_data["bbox"][1]),
                                "bottom_left": (annotation_data["bbox"][0], annotation_data["bbox"][3]),
                                "bottom_right": (annotation_data["bbox"][2], annotation_data["bbox"][3])
                            }

                            for corner, coord in corners.items():
                                if abs(coord[0] - x) < self.corner_size and abs(coord[1] - y) < self.corner_size:
                                    self.annotation_manager.id = annotation_data["id"]
                                    move_top_left = corner == "top_left"
                                    move_top_right = corner == "top_right"
                                    move_bottom_left = corner == "bottom_left"
                                    move_bottom_right = corner == "bottom_right"
                                    move_pose_point = False
                                    self.temp_bbox_coords = annotation_data["bbox"]
                                    breakout = True
                                    break
                            else:
                                move_top_left = move_top_right = move_bottom_left = move_bottom_right = move_pose_point = False

                moved = False
                self.click_count += 1

        elif event == cv2.EVENT_LBUTTONUP:
            self.click_count = 0
            if moved:
           
                if move_pose_point:
                    for i, keypoint in enumerate(self.temp_keypoints):
                        if keypoint[0] == self.keypoint_type and keypoint[1] == self.keypoint_value:
                            self.temp_keypoints[i][1] = (x, y)
                    info = {
                        "images": {
                            "id": self.img_id,
                            "file_name": self.cv2_img.path,
                            "image_height": self.cv2_img.height,
                            "image_width": self.cv2_img.width
                        },
                        "annotation": {
                            "id": self.annotation_manager.id,
                            "keypoints": self.temp_keypoints,
                            "image_id": self.img_id,
                            "object_id": self.object_id,
                            "iscrowd": 0,
                            "type": "pose",
                            "conf": 1,
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                    }
                    self.annotation_manager.save_to_json(info, "pose")
                    self.drawing_annotations()
                    self.show_image()
                else:
                
                    if self.temp_bbox_coords != None:

                        x1, y1, x2, y2 = self.temp_bbox_coords
                        if move_top_left:
                            x1, y1 = x, y
                        elif move_top_right:
                            x2, y1 = x, y
                        elif move_bottom_left:
                            x1, y2 = x, y
                        elif move_bottom_right:
                            x2, y2 = x, y

                        x1, y1 = max(0, min(x1, self.cv2_img.width)), max(0, min(y1, self.cv2_img.height))
                        x2, y2 = max(0, min(x2, self.cv2_img.width)), max(0, min(y2, self.cv2_img.height))

                        if x1 > x2:
                            x1, x2 = x2, x1
                        if y1 > y2:
                            y1, y2 = y2, y1

                        info = {
                            "images": {
                                "id": self.img_id,
                                "efile_name": self.cv2_img.path,
                                "image_height": self.cv2_img.height,
                                "image_width": self.cv2_img.width
                            },
                            "annotation": {
                                "id": self.annotation_manager.id,
                                "bbox": [x1, y1, x2, y2],
                                "image_id": self.img_id,
                                "object_id": self.object_id,
                                "iscrowd": 0,
                                "area": (x2 - x1) * (y2 - y1),
                                "type": self.bbox_type + " bounding_box",
                                "is_hidden": self.is_hidden,
                                "conf": 1,
                                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                        }
                
                        cv2.rectangle(self.cv2_img.get_image(), (x1, y1), (x2, y2), self.annotation_colors[self.object_id], 2)
                        self.annotation_manager.save_to_json(info, "bbox")
                
                        self.drawing_annotations()
                        self.show_image()

        elif self.click_count == 1:
            moved = True
            if move_top_left or move_top_right or move_bottom_left or move_bottom_right or move_pose_point:
                self.cv2_img.set_image()
                for annotation_file in self.annotation_files:
                    with open(self.video_manager.video_dir + "\\" + annotation_file, 'r') as f:
                        data = json.load(f)

                    for i, annotation_data in enumerate(data["annotations"]):
                        if annotation_data["id"] == self.annotation_manager.id:
                            self.img_id = annotation_data["image_id"]
                            self.object_id = annotation_data["object_id"]
                            self.annotation_manager.id = annotation_data["id"]

                            if annotation_data["type"] == "pose":
                                self.temp_keypoints = annotation_data["keypoints"]
                            else:
                                self.temp_bbox_coords = annotation_data["bbox"]
                                self.bbox_type = annotation_data["type"].split()[0]
                                if self.bbox_type == "detected":
                                    self.bbox_type = "normal"
                                self.is_hidden = annotation_data["is_hidden"]

                            file_to_dump = annotation_file
                            del data["annotations"][i]
                            with open(self.video_manager.video_dir + "\\" + file_to_dump, 'w') as f:
                                json.dump(data, f, indent=4)
                            break

                self.drawing_annotations()

                if move_pose_point:
                    for keypoint in self.temp_keypoints:
                        if keypoint[0] != self.keypoint_type or keypoint[1] != self.keypoint_value:
                            cv2.circle(self.cv2_img.get_image(), (keypoint[1][0], keypoint[1][1]), 5, self.annotation_colors[self.object_id], -1)
                            cv2.putText(self.cv2_img.get_image(), keypoint[0].capitalize(), (keypoint[1][0], keypoint[1][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale - 0.25, self.font_color, self.font_thickness)
                    if self.keypoint_type is not None and self.keypoint_value is not None:
                        cv2.circle(self.cv2_img.get_image(), (x, y), 5, self.annotation_colors[self.object_id], -1)
                        cv2.putText(self.cv2_img.get_image(), self.keypoint_type.capitalize(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale - 0.25, self.font_color, self.font_thickness)
                else:
                    x1, y1, x2, y2 = self.temp_bbox_coords
                    if move_top_left:
                        x1, y1 = x, y
                    elif move_top_right:
                        x2, y1 = x, y
                    elif move_bottom_left:
                        x1, y2 = x, y
                    elif move_bottom_right:
                        x2, y2 = x, y

                    corner_points = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
                    for corner_x, corner_y in corner_points:
                        cv2.rectangle(self.cv2_img.get_image(), (corner_x - self.corner_size // 2, corner_y - self.corner_size // 2), (corner_x + self.corner_size // 2, corner_y + self.corner_size // 2), self.annotation_colors[self.object_id], 2)
                    cv2.rectangle(self.cv2_img.get_image(), (x1, y1), (x2, y2), self.annotation_colors[self.object_id], 2)
                    temp_x = max(x1, x2)
                    temp_y = max(y1, y2)
                    cv2.putText(self.cv2_img.get_image(), str(self.object_id), (temp_x - 20, temp_y - 5), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_color, self.font_thickness)

                self.show_image()


        

    def drawing_bbox(self, event, x, y, flags, param):
        """
        Handle drawing of bounding boxes based on mouse events.

        Args:
            event (int): type of mouse event (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP).
            x (int): x-coordinate of the mouse event
            y (int): y-coordinate of the mouse event
            flags (int): relevant flags passed by OpenCV.
            param (any): additional parameters (not used)

        Global Variables:
            start_x (int): x-coordinate where the bounding box drawing starts
            start_y (int): y-coordinate where the bounding box drawing starts
            end_x (int): x-coordinate where the bounding box drawing ends
            end_y (int): y-coordinate where the bounding box drawing ends
        """

        global start_x, start_y, end_x, end_y
        
    

        if self.click_count == 1:
            self.cv2_img.set_image()
            self.drawing_annotations()
        
            image = self.cv2_img.get_image()
            cv2.rectangle(image, (start_x, start_y), (x, y), self.annotation_colors[self.object_id], 2)

            if self.is_hidden == 1:
                self.text_to_write = f"Bounding Box Mode - Hidden - {self.object_id}"
            else:
                if self.bbox_type == "feces":
                    self.text_to_write = "Bounding Box Mode - Feces"
                else:
                    self.text_to_write = f"Bounding Box Mode - {self.object_id}"
                    
            cv2.putText(image, str(self.object_id), (max(start_x, x) - 20, max(start_y, y) - 5), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_color, self.font_thickness)
            self.show_image()
            
        if event == cv2.EVENT_LBUTTONDOWN:
            self.redo_stack = []
            if self.click_count == 0:
            
                start_x = max(0, min(x, self.cv2_img.width))  
                start_y = max(0, min(y, self.cv2_img.height))   
                self.click_count += 1

        elif event == cv2.EVENT_LBUTTONUP:
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

            corner_points = [(start_x, start_y), (end_x, start_y), (start_x, end_y), (end_x, end_y)]

        
            for corner_x, corner_y in corner_points:
                cv2.rectangle(self.cv2_img.get_image(), (corner_x - self.corner_size//2 , corner_y - self.corner_size//2), (corner_x + self.corner_size//2, corner_y + self.corner_size//2), self.annotation_colors[self.object_id], 2)
                
            
            cv2.rectangle(self.cv2_img.get_image(), (start_x, start_y), (end_x, end_y), self.annotation_colors[self.object_id], 2)
            
            self.show_image()
            self.annotation_manager.save_to_json(info, "bbox")


    def drawing_pose(self, event, x, y, flags, param):
        """
        Handle drawing of pose keypoints based on mouse events.

        Args:
            event (int): type of mouse event (cv2.EVENT_LBUTTONDOWN)
            x (int): x-coordinate of the mouse event
            y (int): y-coordinate of the mouse event
            flags (int): relevant flags passed by OpenCV
            param (any): additional parameters (not used)
        """

        with open(self.video_manager.video_dir + "/pose_annotations.json", 'r') as f:
            data = json.load(f)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.redo_stack = []
            point = (x, y)
            cv2.circle(self.cv2_img.get_image(), (point[0], point[1]), 5, self.annotation_colors[self.object_id], -1)

            cv2.putText(self.cv2_img.get_image(), self.pose_type.capitalize(), (point[0], point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale - 0.25, self.font_color, self.font_thickness)
            to_append = (self.pose_type, (point))    
            for annotation in data["annotations"]:
                if annotation["id"] == self.annotation_manager.id:
                    annotation["keypoints"].append(to_append)
                    annotation["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    break
                
            with open(self.video_manager.video_dir + "/pose_annotations.json", 'w') as f:
                json.dump(data, f, indent = 4)
       
            self.drawing_annotations()
           
            self.show_image()


    

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
        """
        Main part of the code is within this function, handles annotation keypresses
        
        """
        

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
            

            if self.is_detected == False and self.model_manager.model_path != "" and self.model_manager.model_path != None and not isinstance(self.model_manager.model_path, tuple) and self.model_detecting == "On":
                
                self.model_manager.img = self.cv2_img
                self.model_manager.img_path = self.cv2_img.path
                self.model_manager.img_width = self.cv2_img.width
                self.model_manager.img_height = self.cv2_img.height
                self.model_manager.img_id = self.img_id
                self.model_manager.object_id = self.object_id
                self.model_manager.annotation_manager = self.annotation_manager
            
             
                self.model_manager.predicting()
              
    
            prev_img_id = self.img_id - 1
            for annotation_file in self.annotation_files:
                with open(self.video_manager.video_dir + "\\" + annotation_file, 'r') as f:
                    data = json.load(f)

                for image_data in data["images"]:
                    if prev_img_id == image_data["id"]:
                        break
                for annotation_data in data["annotations"]:
                    if prev_img_id == annotation_data["image_id"]:
                        if annotation_data["type"] == "normal bounding_box":
                            info = {
                                "id": self.get_id(self.annotation_files, self.video_manager, "annotations"), 
                                "bbox": annotation_data["bbox"],
                                "image_id": self.img_id,
                                "object_id": annotation_data["object_id"],
                                "iscrowd": annotation_data["iscrowd"],
                                "area": annotation_data["area"],
                                "type": annotation_data["type"],
                                "is_hidden": annotation_data["is_hidden"],
                                "conf": 1,
                                "time": (datetime.now() - timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")
                            }
                            

                        elif annotation_data["type"] == "pose":
                            
                            info = {
                                "id": self.get_id(self.annotation_files, self.video_manager, "annotations"),
                                "keypoints": annotation_data["keypoints"],
                                "image_id": self.img_id,
                                "object_id": annotation_data["object_id"],
                                "iscrowd": annotation_data["iscrowd"],
                                "type": annotation_data["type"],
                                "conf": 1,
                                "time": (datetime.now() - timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")
                            }
                            if info:
                                info_copy = copy.deepcopy(info)
                            
                                del info_copy["id"]
                                del info_copy["time"]

                                found_annotation = False
                                for annotation in data["annotations"]:
                                    annotation_copy = copy.deepcopy(annotation)
                                    del annotation_copy["id"]
                                    del annotation_copy["time"]
                                    if annotation_copy == info_copy:
                                        found_annotation = True
                                        break

                                if not found_annotation:
                                    data["annotations"].append(info)

                            if not any(image["id"] == self.img_id for image in data["images"]):
                                new_image_info = {
                                    "id": self.img_id,
                                    "file_name": self.cv2_img.path,
                                    "image_height": self.cv2_img.height,
                                    "image_width": self.cv2_img.width
                                }
                                data["images"].append(new_image_info)


                            with open(self.video_manager.video_dir + "\\" + annotation_file, 'w') as f:
                                
                                json.dump(data, f, indent=4)
        
            self.drawing_annotations()
            self.text_to_write = None 
            self.show_image()

            self.pyqt_button_window.window_name = self.cv2_img.name


            while True:
                key = cv2.waitKey(1)
             
    
                if key == 27 or cv2.getWindowProperty(self.cv2_img.name, cv2.WND_PROP_VISIBLE) < 1: # "Escape": Exits the program 
                    self.annotation_manager.cleaning()
                

                    sys.exit()

                elif key == ord('e') or self.pyqt_button_window.button_states["editing"]:
             
                    if self.pyqt_button_window.button_states["editing"]:
                        self.pyqt_button_window.button_states["editing"] = False
                        
                        
                    if self.editing_mode == False:
                        self.editing_mode = True
                        self.click_count = 0
                        self.cv2_img.set_image()
                        self.drawing_annotations()
                        self.text_to_write = "Editing"
                        self.show_image()
                        self.pose_mode = False
                        self.bbox_mode = False

                        cv2.setMouseCallback(self.cv2_img.name, self.editing)

                    else:
                        self.editing_mode = False
                        self.text_to_write = None
                        self.cv2_img.set_image()
                        self.drawing_annotations()
                        self.show_image()
                        cv2.setMouseCallback(self.cv2_img.name, self.dummy_function)

                elif key == ord('r') or self.pyqt_button_window.button_states["retrain"]:
                    if self.pyqt_button_window.button_states["retrain"]:
                        self.pyqt_button_window.button_states["retrain"] = False
                    self.model_manager.retrain()
                


                elif key == ord('v') or self.pyqt_button_window.button_states["make video"]: # make video
                    if self.pyqt_button_window.button_states["make video"]:
                        self.pyqt_button_window.button_states["make video"] = False
                    
                    self.video_manager.make_video()

                elif (key == ord('m') or self.pyqt_button_window.button_states["toggle model"]) and self.model_manager.model_path != None: # "M": Turns model detection on or off, as shown on the image
          
                    if self.pyqt_button_window.button_states["toggle model"]:
                        self.pyqt_button_window.button_states["toggle model"] = False
                   

                    
                    self.cv2_img.set_image()
                    self.model_detecting = "Off" if self.model_detecting == "On" else "On"
                    self.drawing_annotations()
                    self.show_image()




                    
                elif key == ord('j') or self.pyqt_button_window.button_states["decrement id"]: # "J": Previous object ID
                    if self.pyqt_button_window.button_states["decrement id"]:
                        self.pyqt_button_window.button_states["decrement id"]
                        


                    
                    self.object_id -= 1 if self.object_id > 1 else 0 
                    self.update_img_with_id()
                

                elif key == ord('b') or self.pyqt_button_window.button_states["bounding box"]: # bbox mode
                    if self.pyqt_button_window.button_states["bounding box"]: #= not self.pyqt_button_window.button_states["bounding box"]
                        self.pyqt_button_window.button_states["bounding box"] = False  # Reset the button state after processing it once
                    
                    if not self.bbox_mode: 
                        self.bbox_mode = True
                        self.click_count = 0
                        self.text_to_write = f"Bounding Box Mode - {self.object_id}"
                        self.pose_mode = False
                        self.editing_mode = False
                        self.bbox_type = "normal"
                        cv2.setMouseCallback(self.cv2_img.name, self.drawing_bbox)  # Enable mouse callback for keypoint placement
            
                    else:
                        self.bbox_mode = False
                        self.bbox_type = "normal"
                        self.is_hidden = 0 
                        self.text_to_write = None
                        cv2.setMouseCallback(self.cv2_img.name, self.dummy_function)

                    self.cv2_img.set_image()
                    self.drawing_annotations()
                    self.show_image()
                
                elif key == ord('p') or self.pyqt_button_window.button_states["pose"]: # pose mode
                    if self.pyqt_button_window.button_states["pose"]:
                        self.pyqt_button_window.button_states["pose"] = False
                        

                    if self.pose_mode == False:
                        self.pose_mode = True
                        self.click_count = 0
                        self.cv2_img.set_image()
                        self.drawing_annotations()
                        self.text_to_write = f"Pose Mode - {self.object_id}"
                        
                        self.show_image()
                    
                        self.bbox_mode = False
                        self.editing_mode = False
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

                   
                elif key == 13 or self.pyqt_button_window.button_states["next image"]:
                    self.redo_stack = []
                    if self.pyqt_button_window.button_states["next image"]: # enter; next image in dataset  
                        self.pyqt_button_window.button_states["next image"] = False
                        
                    #cv2.destroyAllWindows()
                    self.pose_mode = False
                    self.bbox_mode = False
                    self.editing_mode = False
                    self.already_passed = False
                    self.object_id = 1
                    self.show_image()
                    cv2.destroyAllWindows()
                    return

                elif key == 8 or self.pyqt_button_window.button_states["previous image"]:
                    self.redo_stack = []
                    if self.pyqt_button_window.button_states["previous image"]:
                        self.pyqt_button_window.button_states["previous image"] = False # backspace; prev_img 
                    self.show_image()
                    self.handle_prev_img()
                    return

                elif key == ord('d') or self.pyqt_button_window.button_states["delete"]: # delete all annotations for an image
                    if self.pyqt_button_window.button_states["delete"]:
                        self.pyqt_button_window.button_states["delete"] = False
                        
                    for annotation_file in self.annotation_files:
                        with open(self.video_manager.video_dir + "\\" + annotation_file, 'r') as f:
                            data = json.load(f)
                        
                            for annotation in data["annotations"]:
                                if annotation["image_id"] == self.img_id:
                                    self.redo_stack.append(annotation)
                        data["annotations"] = [annotation for annotation in data["annotations"] if annotation["image_id"] != self.img_id]
            
                        with open(self.video_manager.video_dir + "\\" + annotation_file, 'w') as f:
                            json.dump(data, f, indent=4)

                    self.cv2_img.set_image()
                    self.text_to_write = None
                    self.show_image()
                    cv2.setMouseCallback(self.cv2_img.name, self.dummy_function)
                    self.bbox_mode = False
                    self.pose_mode = False
                    self.editing_mode = False
                    self.object_id = 1

                elif key == 89 or self.pyqt_button_window.button_states["redo"]: # no code for ctrl + y, pressing "y" == 86; redo
                    if self.pyqt_button_window.button_states["redo"]:
                        self.pyqt_button_window.button_states["redo"] = False
                    
                    if self.redo_stack:
                        info = self.redo_stack.pop()
                        
                        if 'type' in info.keys():
                        
                            info["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            with open(self.video_manager.video_dir + "\\" + "bbox_annotations.json", 'r') as f:
                                data = json.load(f)
                            
                            data["annotations"].append(info)
                            with open(self.video_manager.video_dir + "\\" + "bbox_annotations.json", 'w') as f:
                                json.dump(data, f, indent=4)

                        else:
                            with open(self.video_manager.video_dir + "\\" + "pose_annotations.json", 'r') as f:
                                data = json.load(f)

                            for i in range(len(data["annotations"])):
                                if data["annotations"][i]["id"] == info["id"]:

                                    data["annotations"][i]["keypoints"].append((info['pos'], info['coords']))
                            
                  
                            with open(self.video_manager.video_dir + "\\" + "pose_annotations.json", 'w') as f:
                                json.dump(data, f, indent=4)

                        self.drawing_annotations()
                        self.show_image()
                   
                elif key == 26 or self.pyqt_button_window.button_states["undo"]: # ctrl + z; undo
                    if self.pyqt_button_window.button_states["undo"]:
                        self.pyqt_button_window.button_states["undo"] = False
                        
                        
                    is_empty = True

                    for annotation_file in self.annotation_files:
                        with open(self.video_manager.video_dir + "\\" + annotation_file, 'r') as f:
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
                        with open(self.video_manager.video_dir + "\\" + annotation_file, 'r') as f:
                            data = json.load(f)
                        
                        for i, annotation in enumerate(data["annotations"]):
                            if annotation["image_id"] == self.img_id:
                                timestamp = datetime.strptime(annotation["time"], "%Y-%m-%d %H:%M:%S")
                                if latest_time is None or timestamp > latest_time:
                                    latest_time = timestamp

            
                    for annotation_file in self.annotation_files:
                        with open(self.video_manager.video_dir + "\\" + annotation_file, 'r') as f:
                            data = json.load(f)
                        
                        for i in range(len(data["annotations"])):
                            timestamp = datetime.strptime(data["annotations"][i]["time"], "%Y-%m-%d %H:%M:%S")
                            if timestamp == latest_time:
                                self.object_id = data["annotations"][i]["object_id"]
                                if data["annotations"][i]["type"] == "pose":
                                    if len(data["annotations"][i]["keypoints"]) != 0:
                                        keypoints_pop = data["annotations"][i]["keypoints"].pop()
                                        temp = {'pos': None,
                                                'coords': None,
                                                'id': None}
                                        
                                        self.redo_stack.append({'pos': keypoints_pop[0], 'coords': keypoints_pop[1], 'id': data['annotations'][i]['id']})
                                        break
                                    else:
                                        self.redo_stack.append(data["annotations"][i])
                                        del data["annotations"][i]
                                        break
                                else:
                                    self.redo_stack.append(data["annotations"][i])
                                    del data["annotations"][i]
                                break
                        
                        with open(self.video_manager.video_dir + "\\" + annotation_file, 'w') as f:
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
                 
                elif key == ord('n') or self.pyqt_button_window.button_states["increment id"]: # next mouse ID
                    if self.pyqt_button_window.button_states["increment id"]:
                        self.pyqt_button_window.button_states["increment id"] = False
              
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
                    ord('2'): ("Neck"),
                    ord('3'): ("Tail"),
                    ord('4'): ("R Hand"),
                    ord('5'): ("L Hand"),
                    ord('6'): ("R Leg"),
                    ord('7'): ("L Leg")
                }

                    for keybind, p_label in pose_options.items():
                        if key == keybind or (self.pyqt_button_window.button_states[p_label.lower()]):
                            if self.pyqt_button_window.button_states[p_label.lower()]:
                                self.pyqt_button_window.button_states[p_label.lower()] = False
                            self.cv2_img.set_image()
                            self.drawing_annotations()
                            self.text_to_write = f"Pose Mode - {p_label} - {self.object_id}"
                            
                            self.show_image()
                            self.pose_type = p_label.lower()
                            cv2.setMouseCallback(self.cv2_img.name, self.drawing_pose)
                


    def run_tool(self):
      
        app = QApplication(sys.argv) 
        
        self.model_manager = ModelManager()
        parser = argparse.ArgumentParser()
        parser.add_argument("--frame_skip", type=int, default=50, help="Number of frames to skip")
        parser.add_argument("--model_path", type=str, default=None, help="Path to the model/weights file")
        parser.add_argument("--clustering", type=bool, default=False, help="True/False to turn on/off clustering of chosen dataset")
        args = parser.parse_args()
        self.frame_skip = args.frame_skip
       
        self.model_manager.model_path = args.model_path

        self.annotation_files = ["bbox_annotations.json", "pose_annotations.json"]
        self.model_manager.annotation_files = self.annotation_files
    
     
        
        screen = screeninfo.get_monitors()[0]  # Assuming you want the primary monitor
        width, height = screen.width, screen.height
        self.screen_center_x = int((width - 700) / 2)
        self.screen_center_y = int((height - 500)/ 2)

        # creating a list of random annotation colors that are the same throughout different runs 
        seed = 41
        random.seed(seed)
        for _ in range(30):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)

            color = (r, g, b)
            self.annotation_colors.append(color)

        self.window_info = {"img_name": None, "coordinates": None, "dimensions": None}

        textSize, baseline = cv2.getTextSize("test", cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_thickness)
        textSizeWidth, self.textSizeHeight = textSize

        self.video_manager = VideoManager(self.frame_skip, self.annotation_colors, self.annotation_files)
      
        video_name = self.video_manager.extract_frames()
        self.image_dir = "used_videos/" + video_name.split(".")[0] + "/extracted_frames/"


        # initialize the json files in the respective video directory
        for annotation_file in self.annotation_files:
            if not os.path.exists(self.video_manager.video_dir + "\\" + annotation_file):
                json_content = {"images": [], "annotations": []}
                
                with open(self.video_manager.video_dir + "\\" + annotation_file, 'w') as f:
                    json.dump(json_content, f, indent=4)
        dir_list = None
   
        # if a model is selected, otherwise let the user annotate with no model assistance
        if not isinstance(self.model_manager.model_path, tuple) and self.model_manager.model_path != "" and self.model_manager.model_path != None:
            import torch
            from ultralytics import YOLO
            from clustering import initialize_clustering

            print("CUDA available?: ", torch.cuda.is_available())
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
         
            self.model_manager.model = YOLO(self.model_manager.model_path)
            self.model_manager.model.to(device)
            
            self.model_manager.video_manager = self.video_manager
            self.model_manager.predict_all(self.image_dir)
            self.model_detecting = "On"

            #comment the below code to turn off/on clustering 
            if args.clustering:
                initialize_clustering((self.image_dir), self.model_manager.model_path, self.frame_skip)
                dir_list = os.listdir("used_videos/" + video_name.split(".")[0] + "/clusters/")
                for i, dir in enumerate(dir_list):
                    dir_list[i] = "used_videos/" + video_name.split(".")[0] + "/clusters/" + dir + "/" 
                # delete extracted_frames to save space only if clustering 
                if dir_list:
                    shutil.rmtree(self.image_dir, ignore_errors=True)
        else:
            dir_list = None
            self.model_detecting = "Off"
        self.already_passed = False
        self.object_id = 1
        self.annotations_exists = False
        

        for annotation_file in self.annotation_files:
            with open(self.video_manager.video_dir + "\\" + annotation_file, 'r') as f:
                data = json.load(f)

            if data["annotations"]:
                self.annotations_exists = True
                break

        # Create a window pop up to determine if user wants to continue where they left off 
        result = "no"
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
        self.pyqt_button_window.show()
        for self.current_dir in directories:
            
            self.imgs = os.listdir(self.current_dir)
            

            with tqdm(total=len(self.imgs), desc=f" {(self.current_dir.split('_')[-1]).replace('/', '')}") as pbar:
                while self.img_num < len(self.imgs):
                    
                    self.is_hidden = 0
                    self.annotations_exists = False
                    annotated_image_ids = set()
                    self.img_num = int(self.img_num)
                    imagepath = os.path.join(self.current_dir, self.imgs[int(self.img_num)])
                    imagename = os.path.basename(imagepath)
                    self.cv2_img = CV2Image(imagepath, imagename)
                    self.pyqt_button_window.window_name = self.cv2_img.name
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
        sys.exit(app.exec_()) 



if __name__ == "__main__":
  
    app = QApplication(sys.argv) 
    pyqt_window = ButtonWindow()

    #pyqt_window.show()

    tool = AnnotationTool()

    tool.run_tool()
 
    sys.exit(app.exec_())


  
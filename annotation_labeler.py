import cv2
import os
from pathlib import Path
import shutil
import json
import random
from datetime import datetime
import argparse
import sys
import screeninfo
from tkinter import Tk, filedialog, messagebox
from tqdm import tqdm
import pywinctl as pwc
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from clustering import *
from VideoManager import *
from AnnotationManager import *
from ModelManager import * 
from PyQtWindow import *




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
        self.font_thickness = 1
        self.font_color = (255, 255, 0)
        self.window_info = {"img_name": None, "coordinates": None, "dimensions": None}
   
        self.editing_mode = False

        self.bbox_mode = False
        self.pose_mode = False
        self.bbox_type = None
        self.pyqtwindow = PyQtWindow()
        self.pose_type = None
        self.current_dir = None
        self.imgs = None 
        self.pyqtwindow.setWindowFlags(Qt.WindowStaysOnTopHint)
       
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
        self.pyqtwindow.move_to_coordinates(x - 200, y)

        if self.text_to_write:
            cv2.putText(self.cv2_img.get_image(), self.text_to_write, (int(self.cv2_img.width * 0.05), self.cv2_img.height - int(self.cv2_img.height * 0.05) - self.textSizeHeight), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_color, self.font_thickness)

        cv2.putText(self.cv2_img.get_image(), f"Model: {self.model_detecting}", (int(self.cv2_img.width * 0.75), self.cv2_img.height - int(self.cv2_img.height * 0.05) - self.textSizeHeight), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_color, self.font_thickness)
        
        cv2.imshow(self.cv2_img.name, self.cv2_img.get_image())
        cv2.setWindowProperty(self.cv2_img.name, cv2.WND_PROP_TOPMOST, 1)
    
        
    @staticmethod
    def move_to(window_name, x, y):
        cv2.moveWindow(window_name, x, y)

    def handle_prev_img(self):

        self.bbox_mode = False
        self.pose_mode = False
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
                   
                        corner_points = [(annotation["bbox"][0], annotation["bbox"][1]), (annotation["bbox"][2], annotation["bbox"][1]), (annotation["bbox"][0], annotation["bbox"][3]), (annotation["bbox"][2], annotation["bbox"][3])]
                        for corner_x, corner_y in corner_points:
                            cv2.rectangle(self.cv2_img.get_image(), (corner_x - self.corner_size//2 , corner_y - self.corner_size//2), (corner_x + self.corner_size//2, corner_y + self.corner_size//2), self.annotation_colors[annotation["object_id"]], 2)
                        cv2.rectangle(self.cv2_img.get_image(), (annotation["bbox"][0], annotation["bbox"][1]), (annotation["bbox"][2], annotation["bbox"][3]), self.annotation_colors[annotation["object_id"]], 2)
                        cv2.putText(self.cv2_img.get_image(), str(annotation["object_id"]), (annotation["bbox"][2] - 20, annotation["bbox"][3] - 5), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_color, self.font_thickness)
                        if annotation["type"] == "detected bounding_box":
                            cv2.putText(self.cv2_img.get_image(), f"{annotation['conf']:.2f}", (annotation["bbox"][0], annotation["bbox"][3]), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_color, self.font_thickness)
                    elif annotation_type == "pose":
                        for keypoint_annotation in annotation["keypoints"]: 
                    
                            if keypoint_annotation[1][0] != None or keypoint_annotation[1][1] != None:
                                cv2.circle(self.cv2_img.get_image(), (keypoint_annotation[1][0], keypoint_annotation[1][1]), 5, self.annotation_colors[annotation["object_id"]], -1)
                                cv2.putText(self.cv2_img.get_image(), keypoint_annotation[0].capitalize(), (keypoint_annotation[1][0], keypoint_annotation[1][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_color, self.font_thickness)

    def editing(self, event, x, y, flags, param):
        global move_top_left, move_top_right, move_bottom_right, move_bottom_left, move_pose_point, file_to_dump

        if event == cv2.EVENT_LBUTTONDOWN:
            breakout = False
            if self.click_count == 0:
                for annotation_file in self.annotation_files:
                    if breakout:
                        break
                    with open(self.video_manager.video_dir + "\\" + annotation_file, 'r') as f:
                        data = json.load(f)

                    for image_data in data["images"]:
                        if image_data["file_name"] == self.cv2_img.path:
                            img_id = image_data["id"]
                    
                    for annotation_data in data["annotations"]:

                        if annotation_data["image_id"] == img_id:
                            if annotation_data["type"] == "pose":
                                if len(annotation_data["keypoints"]) == 0:
                                    continue
                                
                                self.temp_keypoints = annotation_data["keypoints"] 
                                
                                move_pose_point = True
                                move_top_left = False
                                move_top_right = False
                                move_bottom_left = False
                                move_bottom_right = False

                                for keypoint in self.temp_keypoints:
                                    if keypoint[1][0] - 5 < x < keypoint[1][0] + 5 and keypoint[1][1] - 5 < y < keypoint[1][1] + 5:
                         
                                        self.keypoint_type = keypoint[0]
                                        self.keypoint_value = keypoint[1] 
                                        
                                        self.annotation_manager.id = annotation_data["id"]
                                        breakout = True
                                        break
                                    else:
                                        self.keypoint_type = None
                                        self.keypoint_value = None
                                
                            else:
                        
                                top_left_coord = (annotation_data["bbox"][0], annotation_data["bbox"][1])
                                top_right_coord = (annotation_data["bbox"][2], annotation_data["bbox"][1])
                                bottom_left_coord = (annotation_data["bbox"][0], annotation_data["bbox"][3])
                                bottom_right_coord = (annotation_data["bbox"][2], annotation_data["bbox"][3])
                                if top_left_coord[0] - self.corner_size < x < top_left_coord[0] + self.corner_size and top_left_coord[1] - self.corner_size < y < top_left_coord[1] + self.corner_size:
                                    self.annotation_manager.id = annotation_data["id"]
                                    move_top_left = True
                                    move_top_right = False
                                    move_bottom_left = False
                                    move_bottom_right = False
                                    move_pose_point = False
                    
                                    breakout = True
                                    break
                                
                                elif top_right_coord[0] - self.corner_size < x < top_right_coord[0] + self.corner_size and top_right_coord[1] - self.corner_size < y < top_right_coord[1] + self.corner_size:
                                    self.annotation_manager.id = annotation_data["id"]
                                    move_top_left = False
                                    move_top_right = True
                                    move_bottom_left = False
                                    move_bottom_right = False
                                    move_pose_point = False
                                    breakout = True
                                    break

                                elif bottom_left_coord[0] - self.corner_size < x < bottom_left_coord[0] + self.corner_size and bottom_left_coord[1] - self.corner_size < y < bottom_left_coord[1] + self.corner_size:
                                    self.annotation_manager.id = annotation_data["id"]
                                
                                    move_top_left = False
                                    move_top_right = False
                                    move_bottom_left = True
                                    move_bottom_right = False
                                    move_pose_point = False
                                    breakout = True
                                    break

                                elif bottom_right_coord[0] - self.corner_size < x < bottom_right_coord[0] + self.corner_size and bottom_right_coord[1] - self.corner_size < y < bottom_right_coord[1] + self.corner_size:
                                    self.annotation_manager.id = annotation_data["id"]
                                    move_top_left = False
                                    move_top_right = False
                                    move_bottom_left = False
                                    move_bottom_right = True
                                    move_pose_point = False
                                    breakout = True
                                    break
                                else:
                                    move_top_left = False
                                    move_top_right = False
                                    move_bottom_left = False
                                    move_bottom_right = False
                                    move_pose_point = False
                                    
                self.click_count += 1


         
        elif event == cv2.EVENT_LBUTTONUP:
            self.click_count = 0

            if move_pose_point:
                for i, keypoint in enumerate(self.temp_keypoints):
                    if keypoint[0] == self.keypoint_type and keypoint[1] == self.keypoint_value:
                        #keypoint[1] = (x, y)
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
                        "image_id":self.img_id,
                        "object_id":self.object_id,
                        "iscrowd": 0,
                        "type": "pose",
                        "is_hidden": self.is_hidden,
                        "conf": 1,
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                    }
                }
        
                self.show_image()
                self.annotation_manager.save_to_json(info, "pose")
            elif move_top_left:
                top_left_x, top_left_y = x, y 
                bottom_right_x, bottom_right_y = self.temp_bbox_coords[2], self.temp_bbox_coords[3]
                top_left_x, top_left_y = max(0, min(top_left_x, self.cv2_img.width)), max(0, min(top_left_y, self.cv2_img.height))
           

                if top_left_x > bottom_right_x:
       
                    bottom_right_x, top_left_x = top_left_x, bottom_right_x

                if top_left_y > bottom_right_y:
                    bottom_right_y, top_left_y = top_left_y, bottom_right_y

                info = {
                    "images": {
                    "id": self.img_id,
                    "file_name": self.cv2_img.path,
                    "image_height": self.cv2_img.height,
                    "image_width": self.cv2_img.width
                    },
                    "annotation": {
                        "id": self.annotation_manager.id,
                        "bbox": [top_left_x, top_left_y, bottom_right_x, bottom_right_y],
                        "image_id":self.img_id,
                        "object_id":self.object_id,
                        "iscrowd": 0,
                        "area": (bottom_right_x - top_left_x) * (bottom_right_y - top_left_y),
                        "type": self.bbox_type + " " + "bounding_box",
                        "is_hidden": self.is_hidden,
                        "conf": 1,
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                    }
                }
            
        
                cv2.rectangle(self.cv2_img.get_image(), (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), self.annotation_colors[self.object_id], 2)
                self.show_image()
                self.annotation_manager.save_to_json(info, "bbox")
            elif move_top_right:
                top_left_x, top_left_y = self.temp_bbox_coords[0], y 
                bottom_right_x, bottom_right_y = x, self.temp_bbox_coords[3]
                top_left_x, top_left_y = max(0, min(top_left_x, self.cv2_img.width)), max(0, min(top_left_y, self.cv2_img.height))
           

                if top_left_x >  bottom_right_x:
       
                     bottom_right_x, top_left_x = top_left_x,  bottom_right_x

                if top_left_y > bottom_right_y:
                    bottom_right_y, top_left_y = top_left_y, bottom_right_y

                info = {
                    "images": {
                    "id": self.img_id,
                    "file_name": self.cv2_img.path,
                    "image_height": self.cv2_img.height,
                    "image_width": self.cv2_img.width
                    },
                    "annotation": {
                        "id": self.annotation_manager.id,
                        "bbox": [top_left_x, top_left_y,  bottom_right_x, bottom_right_y],
                        "image_id":self.img_id,
                        "object_id":self.object_id,
                        "iscrowd": 0,
                        "area": (bottom_right_x - top_left_x) * (bottom_right_y - top_left_y),
                        "type": self.bbox_type + " " + "bounding_box",
                        "is_hidden": self.is_hidden,
                        "conf": 1,
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                    }
                }
            
        
                cv2.rectangle(self.cv2_img.get_image(), (top_left_x, top_left_y), ( bottom_right_x, bottom_right_y), self.annotation_colors[self.object_id], 2)
         
                self.show_image()
                self.annotation_manager.save_to_json(info, "bbox")

            elif move_bottom_left:
                

                top_left_x, top_left_y = x, self.temp_bbox_coords[1]
                bottom_right_x, bottom_right_y = self.temp_bbox_coords[2], y
                #top_left_x, top_left_y = max(0, min(top_left_x, self.cv2_img.width)), max(0, min(top_left_y, self.cv2_img.height))
                if top_left_x >  bottom_right_x:
       
                     bottom_right_x, top_left_x = top_left_x,  bottom_right_x

                if top_left_y > bottom_right_y:
                    bottom_right_y, top_left_y = top_left_y, bottom_right_y

                info = {
                    "images": {
                    "id": self.img_id,
                    "file_name": self.cv2_img.path,
                    "image_height": self.cv2_img.height,
                    "image_width": self.cv2_img.width
                    },
                    "annotation": {
                        "id": self.annotation_manager.id,
                        "bbox": [top_left_x, top_left_y,  bottom_right_x, bottom_right_y],
                        "image_id":self.img_id,
                        "object_id":self.object_id,
                        "iscrowd": 0,
                        "area": (bottom_right_x - top_left_x) * (bottom_right_y - top_left_y),
                        "type": self.bbox_type + " " + "bounding_box",
                        "is_hidden": self.is_hidden,
                        "conf": 1,
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                    }
                }
            
        
                cv2.rectangle(self.cv2_img.get_image(), (top_left_x, top_left_y), ( bottom_right_x, bottom_right_y), self.annotation_colors[self.object_id], 2)
         
                self.show_image()
                self.annotation_manager.save_to_json(info, "bbox")

            elif move_bottom_right:
                top_left_x, top_left_y = self.temp_bbox_coords[0], self.temp_bbox_coords[1]
                
                bottom_right_x, bottom_right_y = x, y

               # top_left_x, top_left_y = max(0, min(top_left_x, self.cv2_img.width)), max(0, min(top_left_y, self.cv2_img.height))
                if top_left_x >  bottom_right_x:
       
                    bottom_right_x, top_left_x = top_left_x,  bottom_right_x

                if top_left_y > bottom_right_y:
                    bottom_right_y, top_left_y = top_left_y, bottom_right_y

                info = {
                    "images": {
                    "id": self.img_id,
                    "file_name": self.cv2_img.path,
                    "image_height": self.cv2_img.height,
                    "image_width": self.cv2_img.width
                    },
                    "annotation": {
                        "id": self.annotation_manager.id,
                        "bbox": [top_left_x, top_left_y,  bottom_right_x, bottom_right_y],
                        "image_id":self.img_id,
                        "object_id":self.object_id,
                        "iscrowd": 0,
                        "area": (bottom_right_x - top_left_x) * (bottom_right_y - top_left_y),
                        "type": self.bbox_type + " " + "bounding_box",
                        "is_hidden": self.is_hidden,
                        "conf": 1,
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                    }
                }
            
        
                cv2.rectangle(self.cv2_img.get_image(), (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), self.annotation_colors[self.object_id], 2)
         
                self.show_image()
                self.annotation_manager.save_to_json(info, "bbox")
                

        
        elif self.click_count == 1:
        

        
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
               

                                file_to_dump = annotation_file
                                del data["annotations"][i]
                                with open(self.video_manager.video_dir + "\\" + file_to_dump, 'w') as f:
                                    json.dump(data, f, indent=4)
                                break
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
                            cv2.putText(self.cv2_img.get_image(), keypoint[0].capitalize(), (keypoint[1][0], keypoint[1][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_color, self.font_thickness)
                    if self.keypoint_type != None and self.keypoint_value != None:
                        cv2.circle(self.cv2_img.get_image(), (x, y), 5, self.annotation_colors[self.object_id], -1)
                        cv2.putText(self.cv2_img.get_image(), self.keypoint_type.capitalize(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_color, self.font_thickness)

                elif move_top_left:
                    x1, y1 = x, y  # Top-left corner of the main rectangle
                    x2, y2 = self.temp_bbox_coords[2], self.temp_bbox_coords[3]  # Bottom-right corner of the main rectangle

                    # Calculate corner rectangles
                

                    corner_points = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]

                    # Draw rectangles for each corner
                    for corner_x, corner_y in corner_points:
                        cv2.rectangle(self.cv2_img.get_image(), (corner_x - self.corner_size//2 , corner_y - self.corner_size//2), (corner_x + self.corner_size//2, corner_y + self.corner_size//2), self.annotation_colors[self.object_id], 2)
                        cv2.rectangle(self.cv2_img.get_image(), (x1, y1), (x2, y2), self.annotation_colors[self.object_id], 2)



                    
                    temp_x = self.temp_bbox_coords[2] if self.temp_bbox_coords[2] > x else x 
                    temp_y = self.temp_bbox_coords[3] if self.temp_bbox_coords[3] > y else y 
               
                    cv2.putText(self.cv2_img.get_image(), str(self.object_id), (temp_x - 20, temp_y - 5), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_color, self.font_thickness)

                elif move_top_right:
                    x1, y1 = self.temp_bbox_coords[0], y  # Top-left corner of the main rectangle
                    x2, y2 = x, self.temp_bbox_coords[3]  # Bottom-right corner of the main rectangle

                    # Calculate corner rectangles


                    corner_points = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]

                    # Draw rectangles for each corner
                    for corner_x, corner_y in corner_points:
                        cv2.rectangle(self.cv2_img.get_image(), (corner_x - self.corner_size//2 , corner_y - self.corner_size//2), (corner_x + self.corner_size//2, corner_y + self.corner_size//2), self.annotation_colors[self.object_id], 2)
                        cv2.rectangle(self.cv2_img.get_image(), (x1, y1), (x2, y2), self.annotation_colors[self.object_id], 2)

                    # cv2.rectangle(self.cv2_img.get_image(), (self.temp_bbox_coords[0], y), (x, self.temp_bbox_coords[3]), self.annotation_colors[self.object_id], 2)

                
                    
                    temp_x = x2 if x2 > x1 else x1 
                    temp_y = y2 if y2 > y1 else y1 
                    
                    cv2.putText(self.cv2_img.get_image(), str(self.object_id), (temp_x - 20, temp_y - 5), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_color, self.font_thickness)

                elif move_bottom_left:
                    x1, y1 = x, self.temp_bbox_coords[1]
                    x2, y2 = self.temp_bbox_coords[2], y


                    corner_points = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]

                    # Draw rectangles for each corner
                    for corner_x, corner_y in corner_points:
                        cv2.rectangle(self.cv2_img.get_image(), (corner_x - self.corner_size//2 , corner_y - self.corner_size//2), (corner_x + self.corner_size//2, corner_y + self.corner_size//2), self.annotation_colors[self.object_id], 2)
                        cv2.rectangle(self.cv2_img.get_image(), (x1, y1), (x2, y2), self.annotation_colors[self.object_id], 2)

                    temp_x = x2 if x2 > x1 else x1 
                    temp_y = y2 if y2 > y1 else y1 
                    
                    cv2.putText(self.cv2_img.get_image(), str(self.object_id), (temp_x - 20, temp_y - 5), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_color, self.font_thickness)


                elif move_bottom_right:
                    x1, y1 = self.temp_bbox_coords[0], self.temp_bbox_coords[1]
                    x2, y2 = x, y
                    corner_points = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]

                    # Draw rectangles for each corner
                    for corner_x, corner_y in corner_points:
                        cv2.rectangle(self.cv2_img.get_image(), (corner_x - self.corner_size//2 , corner_y - self.corner_size//2), (corner_x + self.corner_size//2, corner_y + self.corner_size//2), self.annotation_colors[self.object_id], 2)
                        cv2.rectangle(self.cv2_img.get_image(), (x1, y1), (x2, y2), self.annotation_colors[self.object_id], 2)

                    temp_x = x2 if x2 > x1 else x1 
                    temp_y = y2 if y2 > y1 else y1 
                    
                    cv2.putText(self.cv2_img.get_image(), str(self.object_id), (temp_x - 20, temp_y - 5), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_color, self.font_thickness)


                self.show_image()
    

    def drawing_bbox(self, event, x, y, flags, param):
        global start_x, start_y, end_x, end_y
        
    

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
            
            cv2.putText(self.cv2_img.get_image(), str(self.object_id), (max(start_x, x) - 20, max(start_y, y) - 5), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_color, self.font_thickness)
            self.show_image()
            
        if event == cv2.EVENT_LBUTTONDOWN:
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
            

            if self.is_detected == False and self.model_manager.model_path != "" and self.model_manager.model_path != None and not isinstance(self.model_manager.model_path, tuple) and self.model_detecting == "On":
                
                self.model_manager.img = self.cv2_img
                self.model_manager.img_path = self.cv2_img.path
                self.model_manager.img_width = self.cv2_img.width
                self.model_manager.img_height = self.cv2_img.height
                self.model_manager.img_id = self.img_id
                self.model_manager.object_id = self.object_id
                self.model_manager.annotation_manager = self.annotation_manager
                self.model_manager.video_manager = self.video_manager
             
                self.model_manager.predicting()
              
    
        
            self.drawing_annotations()
            self.text_to_write = None 
            self.show_image()

            self.pyqtwindow.window_name = self.cv2_img.name
            while True:
                key = cv2.waitKey(1)
             
    
                if key == 27 or cv2.getWindowProperty(self.cv2_img.name, cv2.WND_PROP_VISIBLE) < 1: # "Escape": Exits the program 
                    self.annotation_manager.cleaning()
                

                    sys.exit()

                elif key == ord('e') or self.pyqtwindow.button_states["editing"]:
                    if self.pyqtwindow.button_states["editing"]:
                        self.pyqtwindow.button_states["editing"] = False
                        
                        
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
                elif key == ord('r') or self.pyqtwindow.button_states["retrain"]:
                    if self.pyqtwindow.button_states["retrain"]:
                        self.pyqtwindow.button_states["retrain"] = False
                    self.model_manager.retrain()
                


                elif key == ord('v') or self.pyqtwindow.button_states["make video"]: # make video
                    if self.pyqtwindow.button_states["make video"]:
                        self.pyqtwindow.button_states["make video"] = False
                    
                    self.video_manager.make_video()

                elif (key == ord('m') or self.pyqtwindow.button_states["toggle model"]) and self.model_manager.model_path != None: # "M": Turns model detection on or off, as shown on the image
          
                    if self.pyqtwindow.button_states["toggle model"]:
                        self.pyqtwindow.button_states["toggle model"] = False
                   

                    
                    self.cv2_img.set_image()
                    self.model_detecting = "Off" if self.model_detecting == "On" else "On"
                    self.drawing_annotations()
                    self.show_image()




                    
                elif key == ord('j') or self.pyqtwindow.button_states["decrement id"]: # "J": Previous object ID
                    if self.pyqtwindow.button_states["decrement id"]:
                        self.pyqtwindow.button_states["decrement id"]
                        


                    
                    self.object_id -= 1 if self.object_id > 1 else 0 
                    self.update_img_with_id()
                

                elif key == ord('b') or self.pyqtwindow.button_states["bounding box"]: # bbox mode
                    if self.pyqtwindow.button_states["bounding box"]: #= not self.pyqtwindow.button_states["bounding box"]
                        self.pyqtwindow.button_states["bounding box"] = False  # Reset the button state after processing it once
                    
                    if self.bbox_mode == False: 
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
                
                elif key == ord('p') or self.pyqtwindow.button_states["pose"]: # pose mode
                    if self.pyqtwindow.button_states["pose"]:
                        self.pyqtwindow.button_states["pose"] = False
                        

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

                   
                elif key == 13 or self.pyqtwindow.button_states["next image"]:
                    if self.pyqtwindow.button_states["next image"]: # enter; next image in dataset  
                        self.pyqtwindow.button_states["next image"] = False
                        
                    #cv2.destroyAllWindows()
                    self.pose_mode = False
                    self.bbox_mode = False
                    self.already_passed = False
                    self.object_id = 1
                    self.show_image()
                    cv2.destroyAllWindows()
                    return

                elif key == 8 or self.pyqtwindow.button_states["previous image"]:
                    if self.pyqtwindow.button_states["previous image"]:
                        self.pyqtwindow.button_states["previous image"] = False # backspace; prev_img 
                    self.show_image()
                    self.handle_prev_img()
                    return

                elif key == ord('d') or self.pyqtwindow.button_states["delete"]: # delete all annotations for an image
                    if self.pyqtwindow.button_states["delete"]:
                        self.pyqtwindow.button_states["delete"] = False
                        
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

                elif key == 26 or self.pyqtwindow.button_states["undo"]: # ctrl + z; undo
                    if self.pyqtwindow.button_states["undo"]:
                        self.pyqtwindow.button_states["undo"] = False
                        
                        
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
                                        data["annotations"][i]["keypoints"].pop()
                                        break
                                    else:
                                        
                                        del data["annotations"][i]
                                        break
                                else:
                
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
                    
                elif key == ord('n') or self.pyqtwindow.button_states["increment id"]: # next mouse ID
                    if self.pyqtwindow.button_states["increment id"]:
                        self.pyqtwindow.button_states["increment id"] = False
              
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
                    ord('3'): ("Neck"),
                    ord('4'): ("R Hand"),
                    ord('5'): ("L Hand"),
                    ord('6'): ("R Leg"),
                    ord('7'): ("L Leg")
                }

                    for keybind, p_label in pose_options.items():
                        if key == keybind or (self.pyqtwindow.button_states[p_label.lower()] ):
                            self.cv2_img.set_image()
                            self.drawing_annotations()
                            self.text_to_write = f"Pose Mode - {p_label} - {self.object_id}"
                            
                            self.show_image()
                            self.pose_type = p_label.lower()
                            cv2.setMouseCallback(self.cv2_img.name, self.drawing_pose)
                


        
     

    def run_tool(self):
      
        app = QApplication(sys.argv) #
        
        self.model_manager = ModelManager()
        parser = argparse.ArgumentParser()
        parser.add_argument("--frame_skip", type=int, default=50, help="Number of frames to skip")
        parser.add_argument("--model_path", type=str, default=None, help="Path to the model/weights file ")
        args = parser.parse_args()
        self.frame_skip = args.frame_skip
       
        self.model_manager.model_path = args.model_path

        self.annotation_files = ["bbox_annotations.json", "pose_annotations.json"]
        self.model_manager.annotation_files = self.annotation_files
    
     
        
        screen = screeninfo.get_monitors()[0]  # Assuming you want the primary monitor
        width, height = screen.width, screen.height
        self.screen_center_x = int((width - 700) / 2)
        self.screen_center_y = int((height - 500)/ 2)
        # creating a list of random annotation colors that are- the same throughout different runs 
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

            print("CUDA available?: ", torch.cuda.is_available())
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
         
            self.model_manager.model = YOLO(self.model_manager.model_path)
        
            self.model_manager.model.to(device)
            self.model_detecting = "On"

            # comment this to turn off clustering 
            # initialize_clustering(Path(self.image_dir), self.model_manager.model_path)
            # dir_list = os.listdir("used_videos/" + video_name.split(".")[0] + "/clusters/")
            # for i, dir in enumerate(dir_list):
            #     dir_list[i] = "used_videos/" + video_name.split(".")[0] + "/clusters/" + dir + "/" 
            # # delete extracted_frames to save space
            # shutil.rmtree(self.image_dir, ignore_errors=True)
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
        self.pyqtwindow.show()
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
                    self.pyqtwindow.window_name = self.cv2_img.name
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
        sys.exit(app.exec_()) # 



if __name__ == "__main__":


     # Initialize PyQt application and window
    app = QApplication(sys.argv) #


    tool = AnnotationTool()

    
    tool.run_tool()

  
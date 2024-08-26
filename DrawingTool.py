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
from tqdm import tqdm
import pywinctl as pwc
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import Qt
import copy

from ImageHandler import ImageHandler


class DrawingTool():
    def __init__(self):
    
        self.image_handler = ImageHandler()
        self.annotation_files = ["bbox_annotations.json", "pose_annotations.json"]
        self.corner_size = 10


    def dummy_function(self, event, x, y, flags, param):
        pass
        

    def drawing_annotations(self):
        """
        Draw annotations on the current image from the .json files
        """


        for annotation_file in self.annotation_files:
            with open(os.path.join(self.image_handler.video_manager.processed_path, annotation_file), 'r') as f:
        
                annotations = json.load(f)
            
            for annotation in annotations["annotations"]:
                if annotation["image_id"] == self.img_id:
                    
                    # drawing established bbox annotations on the image
                    if annotation["type"].split()[-1] == "bounding_box":
                   
                        corner_points = [(annotation["bbox"][0], annotation["bbox"][1]), (annotation["bbox"][2], annotation["bbox"][1]), (annotation["bbox"][0], annotation["bbox"][3]), (annotation["bbox"][2], annotation["bbox"][3])]
                        for corner_x, corner_y in corner_points:
                            cv2.rectangle(self.image_handler.cv2_img.get_image(), (corner_x - self.corner_size//2 , corner_y - self.corner_size//2), (corner_x + self.corner_size//2, corner_y + self.corner_size//2), self.annotation_colors[annotation["object_id"]], 2)
                        cv2.rectangle(self.image_handler.cv2_img.get_image(), (annotation["bbox"][0], annotation["bbox"][1]), (annotation["bbox"][2], annotation["bbox"][3]), self.annotation_colors[annotation["object_id"]], 2)
                        cv2.putText(self.image_handler.cv2_img.get_image(), str(annotation["object_id"]), (annotation["bbox"][2] - 20, annotation["bbox"][3] - 5), cv2.FONT_HERSHEY_SIMPLEX, self.image_handler.font_scale, self.image_handler.font_color, self.image_handler.font_thickness)
                        if annotation["type"] == "detected bounding_box":
                            cv2.putText(self.image_handler.cv2_img.get_image(), f"{annotation['conf']:.2f}", (annotation["bbox"][0], annotation["bbox"][3]), cv2.FONT_HERSHEY_SIMPLEX, self.image_handler.font_scale, self.image_handler.font_color, self.image_handler.font_thickness)

                    # drawing established pose annotations on the image
                    elif annotation["type"] == "pose":
                        for keypoint_annotation in annotation["keypoints"]: 
                    
                            if keypoint_annotation[1][0] != None or keypoint_annotation[1][1] != None:
                                cv2.circle(self.image_handler.cv2_img.get_image(), (keypoint_annotation[1][0], keypoint_annotation[1][1]), 5, self.annotation_colors[annotation["object_id"]], -1)
                                cv2.putText(self.image_handler.cv2_img.get_image(), keypoint_annotation[0].capitalize(), (keypoint_annotation[1][0], keypoint_annotation[1][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, self.image_handler.font_scale-0.25, self.image_handler.font_color, self.image_handler.font_thickness)
                        
                        # drawing lines correlating to certain pairs
                        if annotation['keypoints']:
                            keypoints = {kp[0]: kp[1] for kp in annotation['keypoints']}
          
                       
                            keypoint_pairs = [
                                ("head", "neck"),
                                ("neck", "tail"),
                                ("neck", "l ear"),
                                ("neck", "r ear"),
                                ("tail", "l leg"),
                                ("tail", "r leg")
                            ]

                            for key1, key2 in keypoint_pairs:
                                if key1 in keypoints and key2 in keypoints:
                                    pt1 = tuple(keypoints[key1])
                                    pt2 = tuple(keypoints[key2])
                                    cv2.line(self.image_handler.cv2_img.get_image(), pt1, pt2, self.annotation_colors[annotation["object_id"]], thickness=1)


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

        # clicking down on the image, initializing which annotation is being chosen to be edited
        if event == cv2.EVENT_LBUTTONDOWN:
            self.temp_bbox_coords = None
            breakout = False
            if self.click_count == 0:
                for annotation_file in self.annotation_files:
                    if breakout:
                        break
                    with open(os.path.join(self.image_handler.video_manager.processed_path, annotation_file), 'r') as f:
                        data = json.load(f)

                    img_id = next((img_data["id"] for img_data in data["images"] if os.path.normpath(img_data["file_name"]) == os.path.normpath(self.image_handler.cv2_img.path)), None)
                    if img_id is None:
                        continue

                    for annotation_data in data["annotations"]:
                        if breakout:
                            break
                        if annotation_data["image_id"] != img_id:
                            continue
                        
                        # selected annotation is pose
                        if annotation_data["type"] == "pose":
                            if len(annotation_data["keypoints"]) == 0:
                                continue

                            self.temp_keypoints = annotation_data["keypoints"]
                            for keypoint in self.temp_keypoints:
                                if abs(keypoint[1][0] - x) < 7.5 and abs(keypoint[1][1] - y) < 7.5:
                                    self.keypoint_type = keypoint[0]
                                    self.keypoint_value = keypoint[1]
                                    self.annotation_manager.id = annotation_data["id"]
                                    
                                    self.object_id = annotation_data["object_id"]
                             
                                    move_pose_point = True
                                    move_top_left = move_top_right = move_bottom_left = move_bottom_right = False
                                    breakout = True
                                    self.editing_stack.append(annotation_data)
                          
                                    break
                                else:
                                    self.keypoint_type = self.keypoint_value = None
                                    move_pose_point = False
                        # selected annotation is a bounding box
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
                                    self.object_id = annotation_data["object_id"]
                                    move_top_left = corner == "top_left"
                                    move_top_right = corner == "top_right"
                                    move_bottom_left = corner == "bottom_left"
                                    move_bottom_right = corner == "bottom_right"
                                    move_pose_point = False
                                    self.temp_bbox_coords = annotation_data["bbox"]
                                    breakout = True
                                    self.editing_stack.append(annotation_data)
                                 
                                    break
                                else:
                                    move_top_left = move_top_right = move_bottom_left = move_bottom_right = move_pose_point = False

                moved = False
                self.click_count += 1
          

        # clicking off of the image, setting the newly edited annotation
        elif event == cv2.EVENT_LBUTTONUP:
            
            self.click_count = 0
            # only if the editing caused the annotation to move
            if moved:
                
                # setting new pose point
                if move_pose_point:
                    for i, keypoint in enumerate(self.temp_keypoints):
                        if keypoint[0] == self.keypoint_type and keypoint[1] == self.keypoint_value:
                            self.temp_keypoints[i][1] = (x, y)
                    info = {
                        "images": {
                            "id": self.img_id,
                            "file_name": self.image_handler.cv2_img.path,
                            "image_height": self.image_handler.cv2_img.height,
                            "image_width": self.image_handler.cv2_img.width
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
         
                    self.image_handler.show_image()
                
                # setting new bbox shape
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

                        x1, y1 = max(0, min(x1, self.image_handler.cv2_img.width)), max(0, min(y1, self.image_handler.cv2_img.height))
                        x2, y2 = max(0, min(x2, self.image_handler.cv2_img.width)), max(0, min(y2, self.image_handler.cv2_img.height))

                        if x1 > x2:
                            x1, x2 = x2, x1
                        if y1 > y2:
                            y1, y2 = y2, y1

                        info = {
                            "images": {
                                "id": self.img_id,
                                "file_name": self.image_handler.cv2_img.path,
                                "image_height": self.image_handler.cv2_img.height,
                                "image_width": self.image_handler.cv2_img.width
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
                
                        cv2.rectangle(self.image_handler.cv2_img.get_image(), (x1, y1), (x2, y2), self.annotation_colors[self.object_id], 2)
                        self.annotation_manager.save_to_json(info, "bbox")
                
                        self.drawing_annotations()
                      
                        self.image_handler.show_image()

        # holding down on the click, moving around and editing the annotation
        elif self.click_count == 1:

            # the annotation has moved
            moved = True

            # can only move pose, or the 4 corners of a bbox
            if move_top_left or move_top_right or move_bottom_left or move_bottom_right or move_pose_point:
                # refresh image
                self.image_handler.cv2_img.set_image()
         
                # retrieving the annotation data
                for annotation_file in self.annotation_files:
                    with open(os.path.join(self.image_handler.video_manager.processed_path, annotation_file), 'r') as f:
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
                            with open(os.path.join(self.image_handler.video_manager.processed_path, file_to_dump), 'w') as f:
                                json.dump(data, f, indent=4)
                            break

                self.drawing_annotations()
                
                # setting new pose point while dragging around
                if move_pose_point:
                    for keypoint in self.temp_keypoints:
                        if keypoint[0] != self.keypoint_type or keypoint[1] != self.keypoint_value:
                            cv2.circle(self.image_handler.cv2_img.get_image(), (keypoint[1][0], keypoint[1][1]), 5, self.annotation_colors[self.object_id], -1)
                            cv2.putText(self.image_handler.cv2_img.get_image(), keypoint[0].capitalize(), (keypoint[1][0], keypoint[1][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, self.image_handler.font_scale - 0.25, self.image_handler.font_color, self.image_handler.font_thickness)
                    if self.keypoint_type is not None and self.keypoint_value is not None:
                        cv2.circle(self.image_handler.cv2_img.get_image(), (x, y), 5, self.annotation_colors[self.object_id], -1)
                        cv2.putText(self.image_handler.cv2_img.get_image(), self.keypoint_type.capitalize(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, self.image_handler.font_scale - 0.25, self.image_handler.font_color, self.image_handler.font_thickness)
                # setting new bbox shape while dragging around
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
                        cv2.rectangle(self.image_handler.cv2_img.get_image(), (corner_x - self.corner_size // 2, corner_y - self.corner_size // 2), (corner_x + self.corner_size // 2, corner_y + self.corner_size // 2), self.annotation_colors[self.object_id], 2)
                    cv2.rectangle(self.image_handler.cv2_img.get_image(), (x1, y1), (x2, y2), self.annotation_colors[self.object_id], 2)
                    temp_x = max(x1, x2)
                    temp_y = max(y1, y2)
                    cv2.putText(self.image_handler.cv2_img.get_image(), str(self.object_id), (temp_x - 20, temp_y - 5), cv2.FONT_HERSHEY_SIMPLEX, self.image_handler.font_scale, self.image_handler.font_color, self.image_handler.font_thickness)
             
                self.image_handler.show_image()


        

    def drawing_bbox(self, event, x, y, flags, param):
        """
        Handle drawing of bounding boxes based on mouse click events.

        Args:
            event (int): type of mouse event (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP).
            x (int): x-coordinate of the mouse event
            y (int): y-coordinate of the mouse event
            flags (int): relevant flags passed by OpenCV
            param (any): additional parameters (not used)

        Global Variables:
            start_x (int): x-coordinate where the bounding box drawing starts
            start_y (int): y-coordinate where the bounding box drawing starts
            end_x (int): x-coordinate where the bounding box drawing ends
            end_y (int): y-coordinate where the bounding box drawing ends
        """

        global start_x, start_y, end_x, end_y
        
    

        if self.click_count == 1:
            self.image_handler.cv2_img.set_image()
            
            self.drawing_annotations()
        
            image = self.image_handler.cv2_img.get_image()
            cv2.rectangle(image, (start_x, start_y), (x, y), self.annotation_colors[self.object_id], 2)

            if self.is_hidden == 1:
                self.image_handler.text_to_write = f"Bounding Box Mode - Hidden - {self.object_id}"
            else:
                if self.bbox_type == "feces":
                    self.image_handler.text_to_write = "Bounding Box Mode - Feces"
                else:
                    self.image_handler.text_to_write = f"Bounding Box Mode - {self.object_id}"
                    
            cv2.putText(image, str(self.object_id), (max(start_x, x) - 20, max(start_y, y) - 5), cv2.FONT_HERSHEY_SIMPLEX, self.image_handler.font_scale, self.image_handler.font_color, self.image_handler.font_thickness)
          
     
            self.image_handler.show_image()
            
        if event == cv2.EVENT_LBUTTONDOWN:
    
            if self.click_count == 0:
            
                start_x = max(0, min(x, self.image_handler.cv2_img.width))  
                start_y = max(0, min(y, self.image_handler.cv2_img.height))   
                self.click_count += 1

        elif event == cv2.EVENT_LBUTTONUP:
            self.annotation_manager.id = self.image_handler.get_id(self.annotation_files, self.image_handler.video_manager, "annotations")
            self.img_id = None
            for annotation_file in self.annotation_files:
                with open(os.path.join(self.image_handler.video_manager.processed_path, annotation_file), 'r') as f:
                    data = json.load(f)

                
                    break_loop = False
                    for image_data in data["images"]:
                        if os.path.normpath(image_data["file_name"]) == os.path.normpath(self.image_handler.cv2_img.path):
                
                            self.img_id = image_data["id"]
                            break_loop = True
                            break
                    if break_loop:
                        break
            if self.img_id == None:
                self.img_id = self.image_handler.get_id(self.annotation_files, self.image_handler.video_manager, "images")
            end_x, end_y = x, y
            self.click_count = 0
                  
            end_x = max(0, min(end_x, self.image_handler.cv2_img.width))
            end_y = max(0, min(end_y, self.image_handler.cv2_img.height))
            if end_x < start_x:
                start_x, end_x = end_x, start_x

            if end_y < start_y:
                start_y, end_y = end_y, start_y

            info = {
                "images": {
                "id": self.img_id,
                "file_name": self.image_handler.cv2_img.path,
                "image_height": self.image_handler.cv2_img.height,
                "image_width": self.image_handler.cv2_img.width
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
                cv2.rectangle(self.image_handler.cv2_img.get_image(), (corner_x - self.corner_size//2 , corner_y - self.corner_size//2), (corner_x + self.corner_size//2, corner_y + self.corner_size//2), self.annotation_colors[self.object_id], 2)
                
            
            cv2.rectangle(self.image_handler.cv2_img.get_image(), (start_x, start_y), (end_x, end_y), self.annotation_colors[self.object_id], 2)
            
          
            self.image_handler.show_image()
            self.annotation_manager.save_to_json(info, "bbox")


    def drawing_pose(self, event, x, y, flags, param):
        """
        Handle drawing of pose keypoints based on mouse click events.

        Args:
            event (int): type of mouse event (cv2.EVENT_LBUTTONDOWN)
            x (int): x-coordinate of the mouse event
            y (int): y-coordinate of the mouse event
            flags (int): relevant flags passed by OpenCV
            param (any): additional parameters (not used)
        """

        with open(os.path.join(self.image_handler.video_manager.processed_path, "pose_annotations.json"), 'r') as f:
            data = json.load(f)

        if event == cv2.EVENT_LBUTTONDOWN:
      
           
            point = (x, y)
            cv2.circle(self.image_handler.cv2_img.get_image(), (point[0], point[1]), 5, self.annotation_colors[self.object_id], -1)

            cv2.putText(self.image_handler.cv2_img.get_image(), self.pose_type.capitalize(), (point[0], point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, self.image_handler.font_scale - 0.25, self.image_handler.font_color, self.image_handler.font_thickness)
            to_append = (self.pose_type, (point))    
            for annotation in data["annotations"]:
                if annotation["id"] == self.annotation_manager.id:
                    annotation["keypoints"].append(to_append)
                    annotation["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    break
                
            with open(os.path.join(self.image_handler.video_manager.processed_path, "pose_annotations.json"), 'w') as f:
                json.dump(data, f, indent = 4)
       
            self.drawing_annotations()

            self.image_handler.show_image()


    def update_img_with_id(self):
        # reread the image but with a new object id and the same titles as before 
 
        if self.bbox_mode == True:
            self.image_handler.cv2_img.set_image()
        
     
            self.drawing_annotations()
            if self.is_hidden == 1:
                self.image_handler.text_to_write = f"Bounding Box Mode - Hidden - {self.object_id}"
            elif self.bbox_type == "feces":
                self.image_handler.text_to_write = f"Bounding Box Mode - Feces"
            elif self.bbox_type == "normal":
                self.image_handler.text_to_write = f"Bounding Box Mode - {self.object_id}"
            
        
            self.image_handler.show_image()

        # initialize a new pose annotation when a new object id is created 
        elif self.pose_mode == True:
          
            self.image_handler.cv2_img.set_image()
      
            self.drawing_annotations()

            pose_mode_text = f"Pose Mode - {self.object_id}"
            if self.pose_type:
                pose_mode_text = f"Pose Mode - {self.pose_type.capitalize()} - {self.object_id}"
                self.annotation_manager.id = self.image_handler.get_id(self.annotation_files, self.image_handler.video_manager, "annotations")
                info = {
                    "images": {
                        "id": self.img_id,
                        "file_name": self.image_handler.cv2_img.path,
                        "image_height": self.image_handler.cv2_img.height,
                        "image_width": self.image_handler.cv2_img.width
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

            self.image_handler.text_to_write = pose_mode_text
        
            self.image_handler.show_image()
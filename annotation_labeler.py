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
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import Qt
import copy
import re
import time

from VideoManager import *
from AnnotationManager import *
from ModelManager import * 
from PyQtWindows import MainWindow
from ImageHandler import CV2Image, ImageHandler
from DrawingTool import DrawingTool




class AnnotationTool():
    def __init__(self):
    
        self.annotation_files = ["bbox_annotations.json", "pose_annotations.json"]
  

        self.annotations_exists = False
        self.is_prev_img = False
        self.current_dir_num = 0
        self.image_dir = None
        self.redo_stack = []
        self.prev_img_annotations = False
    
    

    def annotating(self):
        """
        Main part of the code is within this function, handles annotation keypresses
        
        """
        
        self.drawing_tool.editing_stack = []
        self.drawing_tool.image_handler.cv2_img.set_image()

        self.drawing_tool.bbox_mode = False
        self.drawing_tool.pose_mode = False
        self.editing_mode = False

        self.drawing_tool.object_id = 1
        self.is_detected = False
        self.drawing_tool.is_hidden = 0
        self.drawing_tool.img_id = None
        self.drawing_tool.click_count = 0
        self.prev_img_annotations = False


        # retrieving id of image
        for annotation_file in self.annotation_files:

            with open(os.path.join(self.drawing_tool.image_handler.video_manager.processed_path, annotation_file), 'r') as f:
                data = json.load(f)

                for image_data in data["images"]:
                    if os.path.normpath(image_data["file_name"]) == os.path.normpath(self.drawing_tool.image_handler.cv2_img.path):
                        self.drawing_tool.img_id = image_data["id"]

        if self.drawing_tool.img_id == None:
            self.drawing_tool.img_id = self.drawing_tool.image_handler.get_id(self.annotation_files, self.drawing_tool.image_handler.video_manager, "images")
       

        # if image to annotate already has model predictions then do not predict on it or if model is enabled or not
        with open(os.path.join(self.drawing_tool.image_handler.video_manager.processed_path, "bbox_annotations.json"), 'r') as f:
            data = json.load(f)

            for annotation in data["annotations"]:
                if annotation["image_id"] == self.drawing_tool.img_id:
                    if annotation["type"] == "detected bounding_box":
                        self.is_detected = True
                        break

            if self.is_detected == False and self.model_manager.model_path != "" and self.model_manager.model_path != None and not isinstance(self.model_manager.model_path, tuple) and self.drawing_tool.image_handler.model_detecting == "On":
                
                self.model_manager.img = self.drawing_tool.image_handler.cv2_img
                self.model_manager.img_path = self.drawing_tool.image_handler.cv2_img.path
                self.model_manager.img_width = self.drawing_tool.image_handler.cv2_img.width
                self.model_manager.img_height = self.drawing_tool.image_handler.cv2_img.height
                self.model_manager.img_id = self.drawing_tool.img_id
                self.model_manager.object_id = self.drawing_tool.object_id
                self.model_manager.annotation_manager = self.annotation_manager
                self.model_manager.predicting()
              
            self.annotation_manager.cleaning()

            # continuing the previous image annotations onto the next, to prevent drawing constant annotations
            if not self.is_prev_img:

                for annotation_file in self.annotation_files:
                    with open(os.path.join(self.drawing_tool.image_handler.video_manager.processed_path, annotation_file), 'r') as f:
                        data = json.load(f)
                    self.prev_img_annotations = False

                    if data["images"]:
                        img_index = self.drawing_tool.image_handler.cv2_img.path.rfind('_img_') + 5  
                     
                        jpg_index = self.drawing_tool.image_handler.cv2_img.path.rfind('.jpg')

                    
                        if img_index != -1 and jpg_index != -1:
                
                            number_str = self.drawing_tool.image_handler.cv2_img.path[img_index:jpg_index]
                            number = int(number_str)

                            new_filename = self.drawing_tool.image_handler.cv2_img.path[:img_index] + f'{number - self.frame_skip:05d}' + self.drawing_tool.image_handler.cv2_img.path[jpg_index:]
                            
                        
                            
                            for image_data in data["images"]:
                                
                                if os.path.normpath(new_filename) == os.path.normpath(image_data["file_name"]):
                                    self.prev_img_annotations = True
                                    prev_img_id = image_data["id"]
                                    break
                                
                            if not self.prev_img_annotations:
                                continue


                        for annotation_data in data["annotations"]:
                     
                            if self.drawing_tool.img_id == annotation_data["image_id"]:
                   
                                break
                            if prev_img_id == annotation_data["image_id"]:
                                if annotation_data["type"] == "normal bounding_box":
                                    info = {
                                        "id": self.drawing_tool.image_handler.get_id(self.annotation_files, self.drawing_tool.image_handler.video_manager, "annotations"), 
                                        "bbox": annotation_data["bbox"],
                                        "image_id": self.drawing_tool.img_id,
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
                                        "id": self.drawing_tool.image_handler.get_id(self.annotation_files, self.drawing_tool.image_handler.video_manager, "annotations"),
                                        "keypoints": annotation_data["keypoints"],
                                        "image_id": self.drawing_tool.img_id,
                                        "object_id": annotation_data["object_id"],
                                        "iscrowd": annotation_data["iscrowd"],
                                        "type": annotation_data["type"],
                                        "conf": 1,
                                        "time": (datetime.now() - timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")
                                    }
                                else:
                                    continue

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

                                if not any(image["id"] == self.drawing_tool.img_id for image in data["images"]):
                                    new_image_info = {
                                        "id": self.drawing_tool.img_id,
                                        "file_name": self.drawing_tool.image_handler.cv2_img.path,
                                        "image_height": self.drawing_tool.image_handler.cv2_img.height,
                                        "image_width": self.drawing_tool.image_handler.cv2_img.width
                                    }
                                    data["images"].append(new_image_info)


                                with open(os.path.join(self.drawing_tool.image_handler.video_manager.processed_path, annotation_file), 'w') as f:
                                    
                                    json.dump(data, f, indent=4)
            
            # drawing all possible annotations before drawing any new ones and showing the image
            self.drawing_tool.drawing_annotations()
            self.drawing_tool.image_handler.text_to_write = None
            self.drawing_tool.image_handler.show_image()
            self.drawing_tool.image_handler.pyqt_window.window_name = self.drawing_tool.image_handler.cv2_img.name


            while True:
                key = cv2.waitKey(1)
             
                
                if key == 27 or cv2.getWindowProperty(self.drawing_tool.image_handler.cv2_img.name, cv2.WND_PROP_VISIBLE) < 1: # "Escape": Exits the program 
                    self.annotation_manager.cleaning()
                    sys.exit()


                elif key == ord('e') or self.drawing_tool.image_handler.pyqt_window.button_states["editing"]: # "E": Editing established bounding box and pose annotations
                    self.prev_img_annotations = True
                    if self.drawing_tool.image_handler.pyqt_window.button_states["editing"]:
                        self.drawing_tool.image_handler.pyqt_window.button_states["editing"] = False
                        
                    self.drawing_tool.image_handler.cv2_img.set_image()
                    # toggling on
                    if self.editing_mode == False:
                    
                        self.editing_mode = True
                        self.drawing_tool.click_count = 0
                        self.drawing_tool.image_handler.text_to_write = "Editing"
                        self.drawing_tool.drawing_annotations()
                        
                        self.drawing_tool.image_handler.show_image()
                        self.drawing_tool.pose_mode = False
                        self.drawing_tool.bbox_mode = False

                        cv2.setMouseCallback(self.drawing_tool.image_handler.cv2_img.name, self.drawing_tool.editing)

                    # toggling off 
                    else:
            
                        self.editing_mode = False
                        self.drawing_tool.image_handler.text_to_write = None
                        self.drawing_tool.drawing_annotations()
                        self.drawing_tool.image_handler.show_image()
                        cv2.setMouseCallback(self.drawing_tool.image_handler.cv2_img.name, self.drawing_tool.dummy_function)


                elif key == ord('r') or self.drawing_tool.image_handler.pyqt_window.button_states["retrain"]: # retrain the model by choosing folders with annotated data
                    if self.drawing_tool.image_handler.pyqt_window.button_states["retrain"]:
                        self.drawing_tool.image_handler.pyqt_window.button_states["retrain"] = False
                    self.drawing_tool.image_handler.pyqt_window.close()
                    self.model_manager.retrain()
                


                elif key == ord('v') or self.drawing_tool.image_handler.pyqt_window.button_states["make video"]: # make video of created annotations from selected video folder
                    if self.drawing_tool.image_handler.pyqt_window.button_states["make video"]:
                        self.drawing_tool.image_handler.pyqt_window.button_states["make video"] = False
                    self.drawing_tool.image_handler.video_manager.make_video()


                elif (key == ord('m') or self.drawing_tool.image_handler.pyqt_window.button_states["toggle model"]) and self.model_manager.model_path != None: # "M": Turns model detection on or off, as shown on the image
          
                    if self.drawing_tool.image_handler.pyqt_window.button_states["toggle model"]:
                        self.drawing_tool.image_handler.pyqt_window.button_states["toggle model"] = False
                   

    
                    self.drawing_tool.image_handler.cv2_img.set_image()
              
                    self.drawing_tool.image_handler.model_detecting = "Off" if self.drawing_tool.image_handler.model_detecting == "On" else "On"
                   
                    self.drawing_tool.drawing_annotations()
                    self.drawing_tool.image_handler.show_image()


                elif key == ord('j') or self.drawing_tool.image_handler.pyqt_window.button_states["decrement id"]: # "J": Previous object ID
                    if self.drawing_tool.image_handler.pyqt_window.button_states["decrement id"]:
                        self.drawing_tool.image_handler.pyqt_window.button_states["decrement id"]
                        
               
                    self.drawing_tool.object_id -= 1 if self.drawing_tool.object_id > 1 else 0 
                    self.drawing_tool.image_handler.cv2_img.set_image()
                    self.drawing_tool.update_img_with_id()
                

                elif key == ord('b') or self.drawing_tool.image_handler.pyqt_window.button_states["bounding box"]: # bbox mode
                    self.prev_img_annotations = True
                 
                    if self.drawing_tool.image_handler.pyqt_window.button_states["bounding box"]: 
                        self.drawing_tool.image_handler.pyqt_window.button_states["bounding box"] = False  # Reset the button state after processing it once
                    
                    # toggle on
                    if not self.drawing_tool.bbox_mode: 
                    
                        self.drawing_tool.bbox_mode = True
                        self.drawing_tool.click_count = 0

                        self.drawing_tool.image_handler.text_to_write = f"Bounding Box Mode - {self.drawing_tool.object_id}"
                        self.drawing_tool.pose_mode = False
                        self.editing_mode = False
                  
                        self.drawing_tool.bbox_type = "normal"
                        cv2.setMouseCallback(self.drawing_tool.image_handler.cv2_img.name, self.drawing_tool.drawing_bbox)  # Enable mouse callback for keypoint placement

                    # toggle off 
                    else:
                        
                        self.drawing_tool.bbox_mode = False
                   
                        self.drawing_tool.bbox_type = "normal"
                        self.drawing_tool.is_hidden = 0 
     
                        self.drawing_tool.image_handler.text_to_write = None
                        cv2.setMouseCallback(self.drawing_tool.image_handler.cv2_img.name, self.drawing_tool.dummy_function)

                    self.drawing_tool.image_handler.cv2_img.set_image()
                    self.drawing_tool.drawing_annotations()
     
                    self.drawing_tool.image_handler.show_image()
                
                elif key == ord('p') or self.drawing_tool.image_handler.pyqt_window.button_states["pose"]: # pose mode
                    self.prev_img_annotations = True
                    if self.drawing_tool.image_handler.pyqt_window.button_states["pose"]:
                        self.drawing_tool.image_handler.pyqt_window.button_states["pose"] = False
                        
                    # toggle on
                    if self.drawing_tool.pose_mode == False:
                      
                        self.drawing_tool.pose_mode = True
                        self.drawing_tool.click_count = 0
                        self.drawing_tool.image_handler.cv2_img.set_image()
                        self.drawing_tool.drawing_annotations()

                        self.drawing_tool.image_handler.text_to_write = f"Pose Mode - {self.drawing_tool.object_id}"
                        self.drawing_tool.image_handler.show_image()
                        self.drawing_tool.bbox_mode = False
                        self.editing_mode = False
                        self.drawing_tool.pose_type = ""
                        self.annotation_manager.id = self.drawing_tool.image_handler.get_id(self.annotation_files, self.drawing_tool.image_handler.video_manager, "annotations")
            
                        info = {
                            "images": {
                                "id": self.drawing_tool.img_id,
                                "file_name": self.drawing_tool.image_handler.cv2_img.path,
                                "image_height": self.drawing_tool.image_handler.cv2_img.height,
                                "image_width": self.drawing_tool.image_handler.cv2_img.width
                            },
                            "annotation": {
                                "id": self.annotation_manager.id,
                                "keypoints": [],
                                "image_id":self.drawing_tool.img_id,
                                "object_id":self.drawing_tool.object_id,
                                "iscrowd": 0,
                                "type": "pose",
                                "conf": 1,
                                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                        }
                        self.annotation_manager.save_to_json(info, "pose")
                        cv2.setMouseCallback(self.drawing_tool.image_handler.cv2_img.name, self.drawing_tool.dummy_function)

                    # toggle off
                    else:
                    
           
                        self.drawing_tool.image_handler.text_to_write = None
                        self.drawing_tool.pose_mode = False
                       
                        self.drawing_tool.image_handler.cv2_img.set_image()
                        self.drawing_tool.drawing_annotations()
                        self.drawing_tool.image_handler.show_image()
                        cv2.setMouseCallback(self.drawing_tool.image_handler.cv2_img.name, self.drawing_tool.dummy_function)

                # button to navigate through different clusters if available
                elif self.drawing_tool.image_handler.pyqt_window.cluster_button == True:
              
                    self.current_dir_num = self.drawing_tool.image_handler.pyqt_window.cluster_num
                    
         
                    self.drawing_tool.pose_mode = False
            
                    self.drawing_tool.bbox_mode = False
                 
                    self.editing_mode = False
                 
                    self.is_passed = False
                  
                    self.drawing_tool.object_id = 1
                    cv2.destroyAllWindows()
                    return

                # moving scroll bar will change the image selected
               
                elif self.drawing_tool.image_handler.pyqt_window.scroll_bar_moved == True:
             
                    self.continue_work = False
                    self.drawing_tool.image_handler.img_num = self.drawing_tool.image_handler.pyqt_window.img_num
                    self.drawing_tool.image_handler.pyqt_window.scroll_bar_moved = False
               
                    self.drawing_tool.pose_mode = False
                   
                    self.drawing_tool.bbox_mode = False
             
                    self.editing_mode = False
                
                    self.is_passed = False
               
                    self.drawing_tool.object_id = 1
                    self.drawing_tool.drawing_annotations()
                    self.drawing_tool.image_handler.show_image()

                    self.drawing_tool.image_handler.pyqt_window.scroll_count = 0

                    cv2.destroyAllWindows()
                    return


                elif key == 13 or self.drawing_tool.image_handler.pyqt_window.button_states["next image"]: # enter; next image in dataset  
                    self.drawing_tool.editing_stack = []
                    self.redo_stack = []
                    if self.drawing_tool.image_handler.pyqt_window.button_states["next image"]: 
                        self.drawing_tool.image_handler.pyqt_window.button_states["next image"] = False
              
                    self.is_passed = False
                    self.drawing_tool.object_id = 1
                    self.drawing_tool.image_handler.show_image()
                    cv2.destroyAllWindows()
                    self.drawing_tool.image_handler.pyqt_window.scroll_bar.setValue(self.drawing_tool.image_handler.img_num)
             
                    self.drawing_tool.image_handler.pyqt_window.scroll_bar_moved = False
                    self.is_prev_img = False
                    return

                elif key == 8 or self.drawing_tool.image_handler.pyqt_window.button_states["previous image"]: # backspace; prev_img 
                    self.redo_stack = []
                    self.drawing_tool.editing_stack = []
                    if self.drawing_tool.image_handler.pyqt_window.button_states["previous image"]:
                        self.drawing_tool.image_handler.pyqt_window.button_states["previous image"] = False 

                    self.drawing_tool.image_handler.show_image()
                    self.drawing_tool.image_handler.handle_prev_img()
               
                    self.is_passed = True
                    self.drawing_tool.click_count = 0
                    self.drawing_tool.image_handler.pyqt_window.scroll_bar.setValue(self.drawing_tool.image_handler.img_num)
                    self.drawing_tool.image_handler.pyqt_window.scroll_bar_moved = False

                    self.continue_work = False
                    self.is_prev_img = True
                    return


                elif key == ord('d') or self.drawing_tool.image_handler.pyqt_window.button_states["delete"]: # delete all annotations for an image
                    
                    if self.drawing_tool.image_handler.pyqt_window.button_states["delete"]:
                        self.drawing_tool.image_handler.pyqt_window.button_states["delete"] = False
                        
                    for annotation_file in self.annotation_files:
                        with open(os.path.join(self.drawing_tool.image_handler.video_manager.processed_path, annotation_file), 'r') as f:
                            data = json.load(f)
                        
                            for annotation in data["annotations"]:
                                if annotation["image_id"] == self.drawing_tool.img_id:
                                    self.redo_stack.append(annotation)
                        data["annotations"] = [annotation for annotation in data["annotations"] if annotation["image_id"] != self.drawing_tool.img_id]
            
                        with open(os.path.join(self.drawing_tool.image_handler.video_manager.processed_path, annotation_file), 'w') as f:
                            json.dump(data, f, indent=4)

                
                    self.drawing_tool.image_handler.cv2_img.set_image()
                    
               
                    self.drawing_tool.image_handler.text_to_write = None
                    self.drawing_tool.image_handler.show_image()
                    cv2.setMouseCallback(self.drawing_tool.image_handler.cv2_img.name, self.drawing_tool.dummy_function)
                   
                    self.drawing_tool.bbox_mode = False
                    self.drawing_tool.pose_mode = False
                    self.editing_mode = False
                    self.drawing_tool.object_id = 1
                    self.annotation_manager.cleaning()


                elif key == 89 or self.drawing_tool.image_handler.pyqt_window.button_states["redo"]: # no code for ctrl + y, pressing "y" == 86; redo
                    if self.drawing_tool.image_handler.pyqt_window.button_states["redo"]:
                        self.drawing_tool.image_handler.pyqt_window.button_states["redo"] = False
                    
                    if self.redo_stack:
                        info = self.redo_stack.pop()
                        
                        if 'type' in info.keys():
                        
                            info["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                           
                            with open(os.path.join(self.drawing_tool.image_handler.video_manager.processed_path, "bbox_annotations.json"), 'r') as f:
                                data = json.load(f)
                            
                            data["annotations"].append(info)
                            with open(os.path.join(self.drawing_tool.image_handler.video_manager.processed_path, "bbox_annotations.json"), 'w') as f:
                                json.dump(data, f, indent=4)

                        else:
                            with open(os.path.join(self.drawing_tool.image_handler.video_manager.processed_path, "pose_annotations.json"), 'r') as f:
                                data = json.load(f)

                            for i in range(len(data["annotations"])):
                                if data["annotations"][i]["id"] == info["id"]:

                                    data["annotations"][i]["keypoints"].append((info['pos'], info['coords']))
                            
                  
                            with open(os.path.join(self.drawing_tool.image_handler.video_manager.processed_path, "pose_annotations.json"), 'w') as f:
                                json.dump(data, f, indent=4)

                        self.drawing_tool.drawing_annotations()
                        self.drawing_tool.image_handler.show_image()
                   

                elif key == 26 or self.drawing_tool.image_handler.pyqt_window.button_states["undo"]: # ctrl + z; undo
                    if self.drawing_tool.image_handler.pyqt_window.button_states["undo"]:
                        self.drawing_tool.image_handler.pyqt_window.button_states["undo"] = False
              
                    breakout = False
                    if len(self.drawing_tool.editing_stack) != 0:
                        for annotation_file in self.annotation_files:
                            if breakout:
                                break
                            with open(os.path.join(self.drawing_tool.image_handler.video_manager.processed_path, annotation_file), 'r') as f:
                                data = json.load(f)

                            for i in range(len(data["annotations"])):
                                old_annotation_data = self.drawing_tool.editing_stack.pop()
                          
                                if data["annotations"][i]["id"] == old_annotation_data["id"] and data["annotations"][i]["object_id"] == old_annotation_data["object_id"] and data["annotations"][i]["image_id"] == old_annotation_data["image_id"]:
                                    if data["annotations"][i]["type"] == "pose":
                                        data["annotations"][i]["keypoints"] = old_annotation_data["keypoints"]
                                    else:
                                        
                                        data["annotations"][i]["bbox"] = old_annotation_data["bbox"]
                                        data["annotations"][i]["area"] = old_annotation_data["area"]
                                        data["annotations"][i]["is_hidden"] = old_annotation_data["is_hidden"]

                                    data["annotations"][i]["id"] = old_annotation_data["id"]
                                    data["annotations"][i]["image_id"] = old_annotation_data["image_id"]
                                    data["annotations"][i]["object_id"] = old_annotation_data["object_id"]
                                    data["annotations"][i]["iscrowd"] = old_annotation_data["iscrowd"]
                                    data["annotations"][i]["type"] = old_annotation_data["type"]
                                    data["annotations"][i]["time"] = old_annotation_data["time"]
                                    data["annotations"][i]["conf"] = old_annotation_data["conf"]
                                    breakout = True
                 
                                    with open(os.path.join(self.drawing_tool.image_handler.video_manager.processed_path, annotation_file), 'w') as f:
                                        json.dump(data, f, indent=4)
                                    break
                                else:

                                    self.drawing_tool.editing_stack.append(old_annotation_data)
                             

                        
                        self.drawing_tool.image_handler.cv2_img.set_image()
            
                        self.drawing_tool.drawing_annotations()
                        
                        self.drawing_tool.image_handler.show_image()
                        continue 
                    
                    is_empty = True

                    for annotation_file in self.annotation_files:
                        with open(os.path.join(self.drawing_tool.image_handler.video_manager.processed_path, annotation_file), 'r') as f:
                            data = json.load(f)
            
                        if any(annotation["image_id"] == self.drawing_tool.img_id for annotation in data["annotations"]):
                            is_empty = False
                            break
                    
                    if is_empty:
                        self.drawing_tool.image_handler.show_image()
             
                        self.is_passed = True
                        self.drawing_tool.click_count = 0
                        self.drawing_tool.image_handler.handle_prev_img()
                  
                       
                        self.drawing_tool.object_id = 1
                        return

                    else:
                        latest_time = None
                    
                    for annotation_file in self.annotation_files:
                        with open(os.path.join(self.drawing_tool.image_handler.video_manager.processed_path, annotation_file), 'r') as f:
                            data = json.load(f)
                        
                        for i, annotation in enumerate(data["annotations"]):
                            if annotation["image_id"] == self.drawing_tool.img_id:
                                timestamp = datetime.strptime(annotation["time"], "%Y-%m-%d %H:%M:%S")
                                if latest_time is None or timestamp > latest_time:
                                    latest_time = timestamp

            
                    for annotation_file in self.annotation_files:
                        with open(os.path.join(self.drawing_tool.image_handler.video_manager.processed_path, annotation_file), 'r') as f:
                            data = json.load(f)
                        
                        for i in range(len(data["annotations"])):
                            timestamp = datetime.strptime(data["annotations"][i]["time"], "%Y-%m-%d %H:%M:%S")
                            if timestamp == latest_time:
                                self.drawing_tool.object_id = data["annotations"][i]["object_id"]
                             
                                if data["annotations"][i]["type"] == "pose":
                                    if len(data["annotations"][i]["keypoints"]) != 0:
                                        keypoints_pop = data["annotations"][i]["keypoints"].pop()
                                    
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
                        
                        with open(os.path.join(self.drawing_tool.image_handler.video_manager.processed_path, annotation_file), 'w') as f:
                            json.dump(data, f, indent=4)
            
                    self.drawing_tool.image_handler.cv2_img.set_image()
                    self.drawing_tool.drawing_annotations()
                    # rewriting the previous titles after deletion
                    mode_text = ""
                    if self.drawing_tool.bbox_mode:
                        mode_text = "Bounding Box Mode - "
                        if self.drawing_tool.is_hidden:
                            mode_text += "Hidden - "
                        elif self.drawing_tool.bbox_type == "feces":
                            mode_text += "Feces - "
                      
                        mode_text += str(self.drawing_tool.object_id)
                
                    elif self.drawing_tool.pose_mode:
                        mode_text = "Pose Mode - "
                        if self.drawing_tool.pose_type:
                            mode_text += f"{self.drawing_tool.pose_type.capitalize()} - "
                 
                        mode_text += str(self.drawing_tool.object_id)

                
                   
                    self.is_passed = False
                    self.drawing_tool.drawing_annotations()
                    
             
                    self.drawing_tool.image_handler.text_to_write = mode_text
                    self.drawing_tool.image_handler.show_image()
                 

                elif key == ord('n') or self.drawing_tool.image_handler.pyqt_window.button_states["increment id"]: # next mouse ID
                    if self.drawing_tool.image_handler.pyqt_window.button_states["increment id"]:
                        self.drawing_tool.image_handler.pyqt_window.button_states["increment id"] = False
              
           
                    self.drawing_tool.object_id += 1
                    self.drawing_tool.image_handler.cv2_img.set_image()
                    self.drawing_tool.update_img_with_id()
                    


                # different bbox modes, user can choose to annotate for feces or a bounding box for an object that is hidden
                if self.drawing_tool.bbox_mode:
         
                   
                    bbox_options = {
                        ord('f'): ("feces", "Feces"),
                        ord('h'): ("normal", "Hidden")
                    }

                    for keybind, (bbox_label, mode_message) in bbox_options.items():
                        if key == keybind:
                         
                            self.drawing_tool.image_handler.cv2_img.set_image()
                            self.drawing_tool.drawing_annotations()
                  
            
                            self.drawing_tool.image_handler.text_to_write = f"Bounding Box Mode - {mode_message} - {self.drawing_tool.object_id}"
                            
                            self.drawing_tool.image_handler.show_image() 
                            self.drawing_tool.is_hidden = 1 if bbox_label == "normal" else 0
                           
                            self.drawing_tool.bbox_type = bbox_label.lower()
                            cv2.setMouseCallback(self.drawing_tool.image_handler.cv2_img.name, self.drawing_tool.drawing_bbox)
                    
                # different keypoints of the pose
                elif self.drawing_tool.pose_mode:

                    pose_options = {
                    ord('1'): ("Head"),
                    ord('2'): ("Neck"),
                    ord('3'): ("Tail"),
                    ord('4'): ("R Ear"),
                    ord('5'): ("L Ear"),
                    ord('6'): ("R Leg"),
                    ord('7'): ("L Leg")
                }

                    for keybind, p_label in pose_options.items():
                      
                        if key == keybind or (self.drawing_tool.image_handler.pyqt_window.button_states[p_label.lower()]):
                            if self.drawing_tool.image_handler.pyqt_window.button_states[p_label.lower()]:
                                self.drawing_tool.image_handler.pyqt_window.button_states[p_label.lower()] = False
                           
                            self.drawing_tool.image_handler.cv2_img.set_image()
                            self.drawing_tool.drawing_annotations()
                
                       
                            self.drawing_tool.image_handler.text_to_write = f"Pose Mode - {p_label} - {self.drawing_tool.object_id}"
                            self.drawing_tool.image_handler.show_image()
                            self.drawing_tool.pose_type = p_label.lower()
                            cv2.setMouseCallback(self.drawing_tool.image_handler.cv2_img.name, self.drawing_tool.drawing_pose)
                

    def process_images(self):
        """
        This begins the process to initalize a new CV2Image, and starts to annotate on it depending on certain actions
        """
        self.drawing_tool.is_hidden = 0
        self.annotations_exists = False
        annotated_image_ids = set()
        
        # retrieving new image path and name
        imagepath = os.path.join(self.drawing_tool.image_handler.current_dir, self.drawing_tool.image_handler.imgs[int(self.drawing_tool.image_handler.img_num)])
        imagename = os.path.basename(imagepath)

        # initializing new CV2Image
        self.drawing_tool.image_handler.cv2_img = CV2Image(imagepath, imagename)
        self.drawing_tool.image_handler.video_manager.cv2_img = self.drawing_tool.image_handler.cv2_img
        self.drawing_tool.image_handler.pyqt_window.window_name = self.drawing_tool.image_handler.cv2_img.name


        # only selecting images to annotate that are retrieved from frame skipping
        if int(((self.drawing_tool.image_handler.cv2_img.name.split('_'))[-1]).replace('.jpg', '')) % self.frame_skip == 0:
                        
            
            #self.drawing_tool.image_handler.cv2_img = CV2Image(imagepath, imagename)
            self.drawing_tool.image_handler.video_manager.cv2_img = self.drawing_tool.image_handler.cv2_img

            # retrieves the image ids that have annotations
            annotated_image_ids = self.annotation_manager.cleaning()

            # checks if annotations already exist on the current CV2Image
            if annotated_image_ids and self.is_passed == False:
                for annotation_file in self.annotation_files:
                    
                    with open(os.path.join(self.drawing_tool.image_handler.video_manager.processed_path, annotation_file), 'r') as f:
                        data = json.load(f)

                    if len(data["images"]) == 0:
                        continue

                    for image_data in data["images"]:
                        if os.path.normpath(image_data["file_name"]) == os.path.normpath(self.drawing_tool.image_handler.cv2_img.path):
                            if image_data["id"] in annotated_image_ids:
                                self.annotations_exists = True
                                continue
        

            # if annotations do not exist, always show that image to annotate
            if not self.annotations_exists:
                self.annotating()

            # annotations exist, but user chose not to continue where they left off
            else:
                if self.continue_work == False:
                    self.annotating()

        if self.drawing_tool.image_handler.img_num != len(self.drawing_tool.image_handler.imgs) - 1:
            
            self.drawing_tool.image_handler.img_num += 1

        self.pbar.n = self.drawing_tool.image_handler.img_num
        self.pbar.refresh()


    def run_tool(self):
      
        
        self.drawing_tool = DrawingTool()
        self.drawing_tool.image_handler = ImageHandler()
        self.model_manager = ModelManager()

        # initializing optional parameters
        parser = argparse.ArgumentParser()
        parser.add_argument("--frame_skip", type=int, default=50, help="Number of frames to skip")
        parser.add_argument("--model_path", type=str, default=None, help="Path to the model/weights file")
        parser.add_argument("--clustering", type=str, default="False", help="True/False to turn on/off clustering of chosen dataset")
        args = parser.parse_args()
        self.frame_skip = args.frame_skip
        self.drawing_tool.image_handler.frame_skip = args.frame_skip
        self.model_manager.frame_skip = args.frame_skip
        self.model_manager.model_path = args.model_path

        self.annotation_files = ["bbox_annotations.json", "pose_annotations.json"]
        self.model_manager.annotation_files = self.annotation_files
     
        screen = screeninfo.get_monitors()[0]  # [0] -> primary monitor
        width, height = screen.width, screen.height
        
        # setting the position of the window to be opened to be the middle of the screen
        self.drawing_tool.image_handler.screen_center_x = int((width - 700) / 2)
        self.drawing_tool.image_handler.screen_center_y = int((height - 500)/ 2)

        # creating a list of random annotation colors that are the same throughout different runs 
        seed = 41
        random.seed(seed)
        self.drawing_tool.annotation_colors = []
        for _ in range(30):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            color = (r, g, b)
            self.drawing_tool.annotation_colors.append(color)


        self.drawing_tool.image_handler.window_info = {"img_name": None, "coordinates": None, "dimensions": None}
        self.drawing_tool.image_handler.is_change_set = False
        self.drawing_tool.image_handler.text_to_write = None
    

        # getting the size of text that will be used on the image
        textSize, baseline = cv2.getTextSize("test", cv2.FONT_HERSHEY_SIMPLEX, self.drawing_tool.image_handler.font_scale, self.drawing_tool.image_handler.font_thickness)
        textSizeWidth, self.drawing_tool.image_handler.textSizeHeight = textSize
       
        # initializing VideoManager to extract frames and allow for other tools
        self.drawing_tool.image_handler.video_manager = VideoManager(self.frame_skip, self.drawing_tool.annotation_colors, self.annotation_files)

        # message box for the user to choose if they want to choose a video to annotate or a folder of images
        file_or_video_box = QMessageBox()
        file_or_video_box.setIcon(QMessageBox.Question)
        file_or_video_box.setWindowTitle("Extract frames from a video or select a folder of images")
        file_or_video_box.setText("Do you want to extract frames from a video, or select a folder of frames?")
        
        # Add custom buttons
        file_button = file_or_video_box.addButton("Video File", QMessageBox.AcceptRole)
        folder_button = file_or_video_box.addButton("Image Folder", QMessageBox.RejectRole)
        
        # execute the message box and get the continue_work
        file_or_video_box.exec_()
        
        # determine which button was pressed
        if file_or_video_box.clickedButton() == file_button:
            image_location_name = self.drawing_tool.image_handler.video_manager.extract_frames()
        
        elif file_or_video_box.clickedButton() == folder_button:
            image_location_name = self.drawing_tool.image_handler.video_manager.image_folder_extraction()
        

        self.image_dir = os.path.join("used_folders", image_location_name.split(".")[0], "extracted_frames")
        

        # initialize the json files in the respective video directory
        for annotation_file in self.annotation_files:
            file_path = os.path.join(self.drawing_tool.image_handler.video_manager.processed_path, annotation_file)
            
            # Check if the file exists
            if not os.path.exists(file_path):
                json_content = {"images": [], "annotations": []}
                
                # Create and write json_content to the new file
                with open(file_path, 'w') as f:
                    json.dump(json_content, f, indent=4)
            
            else:
                # If the file exists, check if it's empty
                with open(file_path, 'r+') as f:
                    content = f.read().strip()  # Read and remove any leading/trailing whitespace
                    
                    if not content:  # If the file is empty
                        json_content = {"images": [], "annotations": []}
                        f.seek(0)  # Move the cursor to the beginning of the file
                        json.dump(json_content, f, indent=4)
                        f.truncate()  # Ensure no leftover content is left after the written data



        dir_list = None
   
        # if a model is selected, otherwise let the user annotate with no model assistance
        if not isinstance(self.model_manager.model_path, tuple) and self.model_manager.model_path != "" and self.model_manager.model_path != None:
            import torch
            from ultralytics import YOLO, RTDETR
            from clustering import initialize_clustering

            print("CUDA available?: ", torch.cuda.is_available())
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            data = torch.load(self.model_manager.model_path)
            is_transformer_model = False

          
            # iterate over the dictionary to check for 'RTDETR'
            for key, value in data.items():
                if isinstance(value, dict):  # check if the value is a dictionary
                    if 'RTDETR' in value:
                        print("The model is a transformer-based model (RTDETR).")
                        is_transformer_model = True
                elif 'RTDETR' in str(value):
                    print("The model is a transformer-based model (RTDETR).")
                    is_transformer_model = True
           
            # initializing model
            self.model_manager.model = RTDETR(self.model_manager.model_path) if is_transformer_model else YOLO(self.model_manager.model_path)
            self.model_manager.model.to(device)
            
            # using model to predict on image dir to seperate images to correctly/incorrectly detected
            self.model_manager.video_manager = self.drawing_tool.image_handler.video_manager
            self.model_manager.predict_all(self.image_dir)
    
            self.drawing_tool.image_handler.model_detecting = "On"
       
      
            # if clustering set to True
            if args.clustering and args.clustering != "False":
                initialize_clustering((self.image_dir), self.model_manager.model_path, self.frame_skip, is_transformer_model)
                dir_list = os.listdir("used_folders/" + image_location_name.split(".")[0] + "/clusters/")
                for i, dir in enumerate(dir_list):
                    dir_list[i] = "used_folders/" + image_location_name.split(".")[0] + "/clusters/" + dir + "/" 
                # delete extracted_frames to save space only if clustering 
                if dir_list:
                    shutil.rmtree(self.image_dir, ignore_errors=True)
        else:

            self.drawing_tool.image_handler.model_detecting = "Off"
       
       
        self.is_passed = False
        self.drawing_tool.object_id = 1
        
        
        # seeing if annotations already exist for the video file/image folder being annotated, annotations will be in the json files
        for annotation_file in self.annotation_files:
            with open(os.path.join(self.drawing_tool.image_handler.video_manager.processed_path, annotation_file), 'r') as f:
                data = json.load(f)

            if data["annotations"]:
                self.annotations_exists = True
                break
        
        # creating a message box to ask user if they want to start where they left off
        self.continue_work = False
        if self.annotations_exists:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Question)
            msg_box.setWindowTitle("Continue to Next Image")
            msg_box.setText("Do you want to continue your work on the image following the last annotated image?")
            msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg_box.setDefaultButton(QMessageBox.No)
            
            self.continue_work = msg_box.exec_()

            if self.continue_work == QMessageBox.Yes:
                self.continue_work = True
            else:
                self.continue_work = False
     
        self.drawing_tool.image_handler.img_num = 0
     
        # directory list will be the list of clusters if a model is chosen, or a list of extracted frames
        if not dir_list:

            directories = [self.image_dir]
        else:
            directories = dir_list
            directories = sorted(directories, key=lambda x: int(re.search(r'cluster_(\d+)', x).group(1)))

        # initializing AnnotationManager to manage saving annotations to json files and cleaning json files
        self.annotation_manager = AnnotationManager(self.drawing_tool.image_handler.video_manager.processed_path, self.annotation_files)
        self.drawing_tool.annotation_manager = self.annotation_manager
        
        # initializing the pyqt window of buttons to be next to the CV2 window of the image
        self.drawing_tool.image_handler.pyqt_window = MainWindow()
        self.drawing_tool.image_handler.pyqt_window.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.drawing_tool.image_handler.pyqt_window.show()
   

        has_initialized = False

        # going through clusters or the single directory of all extracted images
        while self.current_dir_num < len(directories):
        
            self.drawing_tool.image_handler.current_dir = directories[self.current_dir_num]
          
            self.drawing_tool.image_handler.imgs = os.listdir(self.drawing_tool.image_handler.current_dir)
            self.drawing_tool.image_handler.pyqt_window.img_list = self.drawing_tool.image_handler.imgs
            if not has_initialized:
                if len(directories) > 1:
                    self.drawing_tool.image_handler.pyqt_window.cluster_count = len(directories)
           
                self.drawing_tool.image_handler.pyqt_window.initialize()
                has_initialized = True

            self.drawing_tool.image_handler.pyqt_window.scroll_bar.setValue(self.drawing_tool.image_handler.img_num)
 
            
            # create a progress bar to see progress of annotations in a cluster or for entire images
            with tqdm(total=len(self.drawing_tool.image_handler.imgs), desc=f" {(self.drawing_tool.image_handler.current_dir.split('_')[-1]).replace('/', '')}") as self.pbar:
                # if no clustering
                if len(directories) == 1:

                    while True:
                        self.process_images()

                # if clustering, 
                else:
                    while self.drawing_tool.image_handler.img_num < len(self.drawing_tool.image_handler.imgs):
                        
                  
                        self.process_images()
             
                        self.drawing_tool.image_handler.pyqt_window.cluster_count = len(directories)

                        if self.drawing_tool.image_handler.pyqt_window.cluster_button == True:
                            self.drawing_tool.image_handler.pyqt_window.cluster_button = False
                            break
        

            # reset img_num for the next directory  
                self.drawing_tool.image_handler.img_num = 0
            self.current_dir_num += 1



if __name__ == "__main__":
  
    app = QApplication(sys.argv) 
    tool = AnnotationTool()
    tool.run_tool()



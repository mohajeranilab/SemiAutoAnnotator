
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


        

class ImageHandler():
    def __init__(self):
        super().__init__()
        self.annotation_files = ["bbox_annotations.json", "pose_annotations.json"]
        
        self.font_scale = 0.5
        self.font_thickness = 0
        self.font_color = (255, 255, 0)
   
      
    def show_image(self): 
        """
        Shows the image, also resizes it to a specific size and also moves it to a specific place on the screen

        *** An issue I ran into is that when creating a new opencv2 window, it creates the window somewhere (previous position of the last opened cv2 window ?)
        *** THEN moves the window to the specified location. This creates an undesired flickering effect.
        """
        global window_width_change, window_height_change
        
        if not any(value is None for value in self.window_info.values()):
            
        
            if pwc.getWindowsWithTitle(self.cv2_img.name):
                window = pwc.getWindowsWithTitle(self.cv2_img.name)[0]
                x, y = window.left, window.top
        
                window_width, window_height = window.width, window.height
        
                if not self.is_change_set:
            
                    window_width_change = window.width - 720
                    window_height_change = window.height - 540
                    self.is_change_set = True
        
                window_width -= window_width_change
                window_height -= window_height_change
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
    
            
                window_width, window_height = 720, 540
        

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
        self.pyqt_window.move_to_coordinates(x - 200, y)
    
        
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

            with open(os.path.join(video_manager.video_dir, annotation_file), 'r') as f:
                data_file = json.load(f)
            id_set.update(data["id"] for data in data_file[data_type])
        id = 0
        while id in id_set:
            id += 1
        return id
    
    
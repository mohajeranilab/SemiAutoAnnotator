
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
from VideoManager import *
from AnnotationManager import *
from ModelManager import * 
from annotation_labeler import * 


from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QComboBox, QPushButton, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt



class PyQtWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.window_name = None


        self.button_states = {
            "bbox": False,
            "pose": False,
            "editing": False,
            "delete": False,
            "undo": False,
            "toggle_model": False,
            "increment_id": False,
            "decrement_id": False,
            "next_img": False,
            "previous_img": False,
            "retrain": False,
            "make_video": False,
            "head": False,
            "tail": False,
            "neck": False,
            "r hand": False,
            "l hand": False,
            "r leg": False,
            "l leg": False
        }

        

        # add the additonal if conditions to the key presses #######################################
        self.setWindowTitle("PyQt Window")
        self.setGeometry(400, 100, 200, 480)  # Set geometry (x, y, width, height)
        self.original_buttons()
       


    def original_buttons(self):

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Create a vertical layout for the central widget
        self.layout = QVBoxLayout(central_widget)
        button_names = [
            "Bounding Box", "Pose", "Editing", "Delete", "Undo", "Toggle Model",
            "Increment ID", "Decrement ID", "Next Image", "Previous Image",
            "Retrain", "Make Video", "Head", "Tail", "Neck", "R Hand", "L Hand", "R Leg", "L Leg"
        ]

 

        self.bbox_button = QPushButton("Bounding Box", self)
        self.bbox_button.setCheckable(True)
        self.bbox_button.clicked.connect(lambda: self.on_button_clicked("bbox"))
        self.layout.addWidget(self.bbox_button)

        self.pose_button = QPushButton("Pose", self)
        self.pose_button.setCheckable(True)
        self.pose_button.clicked.connect(lambda: self.on_button_clicked("pose"))
        self.layout.addWidget(self.pose_button)

        self.editing_button = QPushButton("Editing", self)
        self.editing_button.setCheckable(True)
        self.editing_button.clicked.connect(lambda: self.on_button_clicked("editing"))
        self.layout.addWidget(self.editing_button)

        self.toggle_model_button = QPushButton("Toggle Model", self)
        self.toggle_model_button.setCheckable(True)
        self.toggle_model_button.clicked.connect(lambda: self.on_button_clicked("toggle_model"))
        self.layout.addWidget(self.toggle_model_button)


        for i, name in enumerate(button_names):
            if name not in ["Bounding Box", "Pose", "Editing", "Toggle Model", "Head", "Tail", "Neck", "R Hand", "L Hand", "R Leg", "L Leg"]:
                
                self.button = QPushButton(name, self)
                self.button.setCheckable(False)
                button_state_key = list(self.button_states.keys())[i]  # Get the corresponding key
                self.button.clicked.connect(lambda state, key=button_state_key: self.on_button_clicked(key))
                self.layout.addWidget(self.button)
           

        self.setCentralWidget(central_widget)


    def on_button_clicked(self, key):
        
        

        if key == "pose" and self.button_states[key]:
            self.button_states[key] = not self.button_states[key]
    
            self.original_buttons()
            

        elif key == "pose" and not self.button_states[key]:
            self.button_states[key] = not self.button_states[key]
           
            self.pose_keypoint_buttons()
            # # Clear layout to remove existing buttons
            # for i in reversed(range(self.layout.count())):
            #     layout_item = self.layout.itemAt(i)
            #     if layout_item.widget() and layout_item.widget() != self.pose_button:
            #         layout_item.widget().setParent(None)
            # # Add additional buttons below the Pose button
            # additional_buttons = ["Button A", "Button B", "Button C"]  # Example additional buttons
            # for name in additional_buttons:
            #     self.button = QPushButton(name, self)
            #     self.button.setCheckable(True)
            #     self.layout.addWidget(self.button)
        else:
            self.button_states[key] = not self.button_states[key]


    def pose_keypoint_buttons(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Create a vertical layout for the central widget
        self.layout = QVBoxLayout(central_widget)
        pose_button_names = button_names = [ "Return", "Head", "Tail", "Neck", "R Hand", "L Hand", "R Leg", "L Leg"
        ]
        for i, name in enumerate(button_names):

                
            self.button = QPushButton(name, self)
            self.button.setCheckable(True)
            button_state_key = list(self.button_states.keys())[i]  # Get the corresponding key
            self.button.clicked.connect(lambda state, key=button_state_key: self.on_button_clicked(key))
            self.layout.addWidget(self.button)
           


    def moveEvent(self, event):
        # Capture PyQt window movement event
        super().moveEvent(event)

        # Calculate new position for OpenCV window
        opencv_x = self.pos().x() + 200  # Adjust as needed
        opencv_y = self.pos().y() + 0  # Adjust as needed
        AnnotationTool.move_to(self.window_name, opencv_x, opencv_y)

    def move_to_coordinates(self, x_coord, y_coord):
        self.move(x_coord, y_coord)
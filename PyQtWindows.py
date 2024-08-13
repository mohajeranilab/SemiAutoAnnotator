from PyQt5.QtWidgets import QMainWindow, QPushButton, QWidget, QVBoxLayout, QApplication, QVBoxLayout, QScrollBar
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
import sys
import cv2
import numpy as np
import os


class MainWindow(QMainWindow):
    def __init__(self):
        """
        Initialize PyQtWindow with button states, names and create the main buttons
        """
        
        super().__init__()
        self.window_name = None
        self.alrdy_passed = False
        self.img_num = None
        self.button_states = {
            "bounding box": False,
            "pose": False,
            "editing": False,
            "delete": False,
            "undo": False,
            "redo": False,
            "toggle model": False,
            "increment id": False,
            "decrement id": False,
            "next image": False,
            "previous image": False,
            "retrain": False,
            "make video": False,
            "head": False,
            "tail": False,
            "neck": False,
            "r ear": False,
            "l ear": False,
            "r leg": False,
            "l leg": False
        }
        self.moved = False
        self.cluster_button = False
        self.cluster_count = False


    def initialize(self):
        self.setWindowTitle("Key Presses")
        self.setGeometry(-5000, -5000, 205, 480)  # (x, y, width, height), set to -5000, -5000 so when it is initialized its off the screen
        self.setFixedSize(205, 480)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout(central_widget)

    
        self.original_buttons()



    def clear_layout(self):
        """
        Clears current layout by removing all widgets
        """

        for i in reversed(range(self.layout.count())):
            widget = self.layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()


    def create_static_buttons(self):
        """
        Create minimize, exit, and scrollbar widgets that will always be on the window
        """
        self.adjust_size()
        # create minimize button
        self.minimize_button = QPushButton("Minimize", self)
        self.minimize_button.clicked.connect(lambda: self.showMinimized())
        icon = QIcon("assets/images/minimize.png")
        self.minimize_button.setIcon(icon)
        self.layout.addWidget(self.minimize_button)

        # create exit button
        self.exit_button = QPushButton("Exit", self)
        self.exit_button.clicked.connect(lambda: self.close())
        icon = QIcon("assets/images/exit.png")
        self.exit_button.setIcon(icon)
        self.layout.addWidget(self.exit_button)

        if self.cluster_count:
            self.all_cluster_button = QPushButton(f"Clusters", self)
            self.all_cluster_button.clicked.connect(lambda checked, cluster_count=self.cluster_count: self.create_cluster_buttons(cluster_count))
            self.layout.addWidget(self.all_cluster_button)

        # if self.cluster_count:     
        #     for i in range(self.cluster_count):
        #         self.cluster_button = QPushButton(f"Cluster {i}", self)
        #         self.cluster_button.clicked.connect(lambda checked, index=i: self.cluster_testing(index))
        #         self.layout.addWidget(self.cluster_button)

        # create scrollbar 
        #if hasattr(MainWindow, "img_list"):
        self.scroll_area = QWidget(self)
        self.scroll_layout = QVBoxLayout(self.scroll_area)
        self.scroll_bar = QScrollBar()
        self.scroll_bar.setOrientation(Qt.Horizontal)
        self.scroll_bar.setMinimum(0)
        self.scroll_bar.setMaximum(len(self.img_list) - 1)
        self.scroll_bar.valueChanged.connect(self.on_scroll)
        self.scroll_layout.addWidget(self.scroll_bar)
        self.layout.addWidget(self.scroll_area)
         # Adjust the size of the window to fit the layout
    
    def cluster_testing(self, num):
        self.cluster_button = True
        self.cluster_num = num
   
        

    def create_cluster_buttons(self, cluster_count):
        self.clear_layout()
        for i in range(cluster_count):
            self.cluster_button = QPushButton(f"Cluster {i}", self)
            self.cluster_button.clicked.connect(lambda checked, index=i: self.cluster_testing(index))
            self.layout.addWidget(self.cluster_button)
        

             
        self.return_button = QPushButton("Return", self)
        self.return_button.clicked.connect(self.original_buttons)
        icon = QIcon("assets/images/undo.png")
        self.return_button.setIcon(icon)
        self.layout.addWidget(self.return_button)

        self.create_static_buttons()


    def adjust_size(self):
        """
        Adjust the size of the window based on the content
        """
        # Optionally, you can compute the size based on number of buttons or other content
        content_height = self.layout.sizeHint().height()
        content_width = self.layout.sizeHint().width()
        self.resize(205, content_height)  # Adjust height and widt


    def original_buttons(self):
        """
        Create the main buttons in the PyQtWindow
        """

        if self.alrdy_passed:
            self.clear_layout()
        else:
            self.alrdy_passed = True
       
        
        # reset button states
        for button_function in self.button_states.keys():
            self.button_states[button_function] = False


        # mapping of button names to their display
        button_mappings = {
            "Bounding Box": "Bounding Box (B)",
            "Pose": "Pose (P)",
            "Editing": "Editing (E)",
            "Toggle Model": "Toggle Model (M)",
            "Delete": "Delete (D)",
            "Undo": "Undo (Ctrl + Z)",
            "Redo": "Redo (Ctrl + Y)",
            "Increment ID": "Increment ID (N)",
            "Decrement ID": "Decrement ID (J)",
            "Next Image": "Next Image (Enter)",
            "Previous Image": "Previous Image (Backspace)",
            "Retrain": "Retrain (R)",
            "Make Video": "Make Video (V)"
            
        }
        
        # creating buttons
        for key, text in button_mappings.items():
            self.button = QPushButton(text, self)
            self.button.clicked.connect(lambda state, key=key: self.on_button_clicked(key.lower()))
            icon = QIcon(f"assets/images/{key.lower()}.png")
            self.layout.addWidget(self.button)
            self.button.setIcon(icon)

        self.create_static_buttons()
    
    def on_scroll(self, value):
 
        self.moved = True
        self.img_num = value
      
    def on_button_clicked(self, key):
        """
        Change the states of buttons when clicked

        Params:
            key (str): the "key" will be the button state to update
        """

        self.button_states[key] = not self.button_states[key]
        
        if key == "pose":
            if self.button_states[key]:
                self.pose_keypoint_buttons()
            else:
                self.original_buttons()
        else:
            for k in self.button_states.keys():
                if k != key:
                    self.button_states[k] = False


    def pose_keypoint_buttons(self):
        """
        Creating buttons for pose keypoints
        """

        self.clear_layout()
        
        
        pose_button_names = ["Head", "Neck", "Tail", "R Ear", "L Ear", "R Leg", "L Leg"]

        for i, name in enumerate(pose_button_names):
            self.button = QPushButton(name + f" ({(i+1)})", self)
            self.button.clicked.connect(lambda state, key=name: self.on_button_clicked(key.lower()))
            icon = QIcon(f"assets/images/{name.lower()}.png")
            self.button.setIcon(icon)
            self.layout.addWidget(self.button)
           
        self.return_button = QPushButton("Return", self)
        self.return_button.clicked.connect(self.original_buttons)
        icon = QIcon("assets/images/undo.png")
        self.return_button.setIcon(icon)
        self.layout.addWidget(self.return_button)

        self.create_static_buttons()

    def moveEvent(self, event):
        """
        Override the move event to handle window movement
        """
        from ImageHandler import ImageHandler 
        # from annotation_labeler import ImageHandler 
        super().moveEvent(event)


        if self.window_name:
            opencv_x = self.pos().x() + 200
            opencv_y = self.pos().y()
            
            ImageHandler.move_to(self.window_name, opencv_x, opencv_y)


    def move_to_coordinates(self, x_coord, y_coord):
        """
        Move the window to specified coordinates

        Params:
            x_coord (int): The x-coordinate to move the window to
            y_coord (int): The y-coordinate to move the window to
        """
        
        self.move(x_coord, y_coord)

       

if __name__ == "__main__":
    app = QApplication(sys.argv)
    img_list = os.listdir("used_videos\\f_2024_05_31_12_58_53_05\\extracted_frames\\")
    window = MainWindow(img_list)
    window.show()
    sys.exit(app.exec_())
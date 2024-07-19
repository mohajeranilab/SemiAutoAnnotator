from annotation_labeler import * 

from PyQt5.QtWidgets import QMainWindow, QPushButton, QWidget, QVBoxLayout
from PyQt5.QtGui import QIcon


class PyQtWindow(QMainWindow):
    def __init__(self):
        """
        Initialize PyQtWindow with button states, names and create the main buttons
        """

        super().__init__()
        self.window_name = None


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
            "r hand": False,
            "l hand": False,
            "r leg": False,
            "l leg": False
        }

    
        self.setWindowTitle("Key Presses")
        self.setGeometry(400, 100, 205, 480)  # (x, y, width, height)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout(central_widget)

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

        self.original_buttons()


    def clear_layout(self):
        """
        Clears current layout by removing all widgets
        """

        for i in reversed(range(self.layout.count())):
            widget = self.layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
                

    def original_buttons(self):
        """
        Create the main buttons in the PyQtWindow
        """
      
        self.clear_layout()
        
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
        
        
        pose_button_names = ["Head", "Neck", "Tail", "R Hand", "L Hand", "R Leg", "L Leg"]

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


    def moveEvent(self, event):
        """
        Override the move event to handle window movement
        """
        super().moveEvent(event)

    
        if self.window_name:
            opencv_x = self.pos().x() + 200
            opencv_y = self.pos().y()
            AnnotationTool.move_to(self.window_name, opencv_x, opencv_y)


    def move_to_coordinates(self, x_coord, y_coord):
        """
        Move the window to specified coordinates

        Params:
            x_coord (int): The x-coordinate to move the window to
            y_coord (int): The y-coordinate to move the window to
        """
        self.move(x_coord, y_coord)


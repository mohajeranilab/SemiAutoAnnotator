import cv2
import os
from pathlib import Path
import numpy as np
import random
import json
from datetime import datetime
from tkinter import Tk, filedialog, messagebox
import shutil
import sys
import screeninfo  
import time
import pygetwindow as gw
import argparse

from ultralytics import YOLO 
import torch

from utils import *
from file_operations import *
from clustering import *

def dummy_function(event, x, y, flags, param):
    pass


def clustering_data(image_dir, model_dir):
   main(Path(image_dir), model_dir)
   pass


def show_image(): 
    """
    Shows the image, also resizes it to a specific size and also moves it to a specific place on the screen

    ***Moving to a specific place solved my issue of the cv2 window opening in random parts of the screen 
    """
    global coordinate_stack
   
    if not any(value is None for value in coordinate_stack.values()):
        
            if gw.getWindowsWithTitle(img_name):
               
                window = gw.getWindowsWithTitle(img_name)[0]  # Get the first window with the given title
                x, y = window.left, window.top
                #print(window.width, window.height)
                #window_width, window_height = window.width, window.height
                 
                #if img_name == image_name:
                coordinate_stack = {}
                coordinate_stack["img_name"] = img_name
                coordinate_stack["coordinates"] = (x, y)
                #coordinate_stack["dimensions"] = (window_width, window_height)
                # coordinate_stack = (img_name, ((x, y), (window_width, window_height)))
              
            else:
                # x, y = coordinate_stack[1][0]
                # window_width, window_height = [1][1]
                x, y = coordinate_stack['coordinates']
                #window_width, window_height = coordinate_stack['dimensions']
    else:
        if gw.getWindowsWithTitle(img_name):
          
            window = gw.getWindowsWithTitle(img_name)[0]  # Get the first window with the given title
            x, y = window.left, window.top
           
            window_width, window_height = window.width, window.height

        else:
            x, y = screen_center_x, screen_center_y
   
            window_width, window_height = 640, 480


    #cv2.destroyAllWindows()
    if coordinate_stack['img_name'] != img_name:
        coordinate_stack = {}
        coordinate_stack['img_name'] = img_name
        coordinate_stack['coordinates'] = (x, y)
        #coordinate_stack['dimensions'] = (window_width, window_height)
        #coordinate_stack = (img_name, ((x, y), (window_width, window_height)))
        cv2.destroyAllWindows()
    

    cv2.namedWindow(img_name, cv2.WINDOW_NORMAL)  
    # cv2.resizeWindow(img_name, (window_width, window_height))  
    cv2.resizeWindow(img_name, (700, 500))  
    cv2.moveWindow(img_name, x, y)
    if text_to_write:
        cv2.putText(img, text_to_write, (int(IMAGE_WIDTH * 0.05), IMAGE_HEIGHT - int(IMAGE_HEIGHT * 0.05) - textSizeHeight), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
    cv2.putText(img, f"Model: {model_detecting}", (int(IMAGE_WIDTH * 0.75), IMAGE_HEIGHT - int(IMAGE_HEIGHT * 0.05) - textSizeHeight), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
  
    cv2.imshow(img_name, img)
   

def handle_prev_img():
    """
    Handles the process of going to the previous image by pressing "Backspace" or pressing Ctrl + Z with no annotations made on an image
    """
    global img_num, bbox_mode, click_count, pose_mode, already_passed

    bbox_mode = False
    pose_mode = False
    already_passed = True
    click_count = 0


    if img_num == 0:
        img_num -= 1
        cv2.destroyAllWindows()
        return
    
    img_num -= 1 
    while img_num < len(imgs):
        img_path = os.path.join(current_dir, imgs[img_num])
        img_name = os.path.basename(img_path)
        if int(((img_name.split('_'))[-1]).replace('.jpg', '')) % FRAME_SKIP != 0:

            if img_num > 0:
                img_num -= 1 
      
              
        else:
            img_num -= 1
            cv2.destroyAllWindows() 
            return


def drawing_annotations(img):
    """
    This will draw the existing annotations on the cv2 image

    img (numpy.ndarray): cv2 image
    """
    annotation_types = ["bbox", "pose"]

    for type_idx, annotation_type in enumerate(annotation_types):
        with open(video_extraction_dir + "\\" + ANNOTATION_FILES[type_idx], 'r') as f:
            annotations = json.load(f)
        
        for annotation in annotations["annotations"]:
            if annotation["image_id"] == img_id:
                if annotation_type == "bbox":
                    img = cv2.rectangle(img, (annotation["bbox"][0], annotation["bbox"][1]), (annotation["bbox"][2], annotation["bbox"][3]), ANNOTATION_COLORS[annotation["object_id"]], 2)
                    img = cv2.putText(img, str(annotation["object_id"]), (annotation["bbox"][2] - 10, annotation["bbox"][3]), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
                    if annotation["type"] == "detected bounding_box":
                        cv2.putText(img, f"{annotation['conf']:.2f}", (annotation["bbox"][0], annotation["bbox"][3]), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
                elif annotation_type == "pose":
                    for keypoint_annotation in annotation["keypoints"]: 
                  
                        if keypoint_annotation[1][0] != None or keypoint_annotation[1][1] != None:
                            img = cv2.circle(img, (keypoint_annotation[1][0], keypoint_annotation[1][1]), 5, ANNOTATION_COLORS[annotation["object_id"]], -1)
                            img = cv2.putText(img, keypoint_annotation[0].capitalize(), (keypoint_annotation[1][0], keypoint_annotation[1][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)


def drawing_bbox(event, x, y, flags, param):
    """
    Draws a bounding box on an image, click and drag and let go to finish building the annotation
    Coordinates are (x_topleft, y_topleft), (x_bottomright, y_bottomright)

    Has two other features, feces for drawing bounding boxes around feces, and
    hidden for drawing bounding boxes around objects that are hidden/obscured
    """

    global click_count, start_x, start_y, img, object_id, annotation_id, img_id, text_to_write
    annotation_id = get_id(ANNOTATION_FILES, "annotations", video_extraction_dir)
    img_id = None
    for annotation_file in ANNOTATION_FILES:
        with open(video_extraction_dir + "\\" + annotation_file, 'r') as f:
            data = json.load(f)

            found = False
            break_loop = False
            for image_data in data["images"]:
                if image_data["file_name"] == img_path:
                   
                    img_id = image_data["id"]
                    break_loop = True
                    break
            if break_loop:
                break
    if img_id == None:
        img_id = get_id(ANNOTATION_FILES, "images", video_extraction_dir)
  

    if click_count == 1:
        img = cv2.imread(img_path)
        drawing_annotations(img)

        img = cv2.rectangle(img, (start_x, start_y), (x, y), ANNOTATION_COLORS[object_id], 2)

        if is_hidden == 1:
        
            text_to_write = f"Bounding Box Mode - Hidden - {object_id}"
        elif bbox_type == "feces":
            text_to_write = f"Bounding Box Mode - Feces"
        elif bbox_type == "normal":
            text_to_write = f"Bounding Box Mode - {object_id}"
        img = cv2.putText(img, str(object_id), (x - 10, y), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
        show_image()

    if event == cv2.EVENT_LBUTTONDOWN:
        if click_count == 0:
         
            start_x = max(0, min(x, IMAGE_WIDTH))  
            start_y = max(0, min(y, IMAGE_HEIGHT))   
            click_count += 1


    elif event == cv2.EVENT_LBUTTONUP:
        
            end_x, end_y = x, y
            click_count = 0
                  
            end_x = max(0, min(end_x, IMAGE_WIDTH))
            end_y = max(0, min(end_y, IMAGE_HEIGHT))
            if end_x < start_x:
                start_x, end_x = end_x, start_x

            if end_y < start_y:
                start_y, end_y = end_y, start_y

            info = {
                "images": {
                "id": img_id,
                "file_name": img_path,
                "image_height": IMAGE_HEIGHT,
                "image_width": IMAGE_WIDTH
                },
                "annotation": {
                    "id": annotation_id,
                    "bbox": [start_x, start_y, end_x, end_y],
                    "image_id":img_id,
                    "object_id":object_id,
                    "iscrowd": 0,
                    "area": (end_x - start_x) * (end_y - start_y),
                    "type": bbox_type + " " + "bounding_box",
                    "is_hidden": is_hidden,
                    "conf": 1,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                }
            }
           
            print(f'Bounding Box added: (X1={start_x}, Y1={start_y}, X2={end_x}, Y2={end_y})')
            img = cv2.rectangle(img, (start_x, start_y), (end_x, end_y), ANNOTATION_COLORS[object_id], 2)
            
            show_image()
            save_to_json(info, "bbox", video_extraction_dir, ANNOTATION_FILES)
   
    
def drawing_pose(event, x, y, flags, param):
    """
    
    Draws keypoints (pose) on an image, click on the screen to make the keypoint annotation


    Has three choices, keypress "1" for Head, keypress "2" for Tail, and keypress "3" for Neck
    
    """

    global start_x, start_y, img, object_id, pose_type
    

    with open(video_extraction_dir + "/pose_annotations.json", 'r') as f:
        data = json.load(f)

    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        img = cv2.circle(img, (point[0], point[1]), 5, ANNOTATION_COLORS[object_id], -1)

        img = cv2.putText(img, pose_type.capitalize(), (point[0], point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
        
        show_image()
        to_append = (pose_type, (point))    
        for annotation in data["annotations"]:
            if annotation["id"] == annotation_id:
                annotation["keypoints"].append(to_append)
                annotation["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                break
            
        with open(video_extraction_dir + "/pose_annotations.json", 'w') as f:
            json.dump(data, f, indent = 4)
                     

def retrain():
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
        
        assert os.path.exists(os.path.join(video_path, "bbox_annotations.json")), \
            f"There are no annotations made for {video_path}"
        
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
                        width = (annotation["bbox"][2] - annotation["bbox"][0])/IMAGE_WIDTH

                        height = (annotation["bbox"][3] - annotation["bbox"][1])/IMAGE_HEIGHT
                        x_center = (annotation["bbox"][0]/IMAGE_WIDTH) + width/2
                        y_center = (annotation["bbox"][1]/IMAGE_HEIGHT) + height/2

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

    
    model.train(data="annotation_labeler.yaml", epochs = 100, patience=15, degrees=30, shear = 30)
    print("Model has finished training, use the new model weights and run the program again.")
    sys.exit()



def add_num_to_detections(event, x, y, flags, param):
    global keep_processing

    with open(video_extraction_dir + "/bbox_annotations.json", 'r') as f:
        data = json.load(f)
    # Update the annotations to reflect the new object ID
        for annotation in data["annotations"]:
            if annotation["image_id"] == img_id and annotation["object_id"] != 0:
                
                keep_processing = False  # Set the flag to False to break the loop
     
                return  # Exit the for loop

    if event == cv2.EVENT_LBUTTONDOWN:

        center_x, center_y = x, y
     
        detected_annotations = [] 
        for annotation in data["annotations"]:
      
            if annotation["image_id"] == img_id and annotation["type"] == "detected bounding_box":
                detected_annotations.append(annotation["bbox"])
        distance_to_point = float('inf')
        for detected_annotation in detected_annotations:
        
            # Midpoint coordinates
            midpoint = ((detected_annotation[0] + detected_annotation[2]) / 2, (detected_annotation[1] + detected_annotation[3]) / 2)
            
            if distance_to_point > euclidean_distance(midpoint, (center_x, center_y)):
                distance_to_point = euclidean_distance
                corresponding_bbox = detected_annotation
        
        for annotation in data["annotations"]:
            if annotation["image_id"] == img_id and annotation["type"] == "detected bounding_box" and annotation["bbox"] == corresponding_bbox:
                annotation["object_id"] = object_id

        with open(video_extraction_dir + "/bbox_annotations.json", 'w') as f:
            json.dump(data, f, indent=4)


     


def annotating(img_path, img_name, video_extraction_dir):
    """
    Main annotation function 
    """

    global already_passed, bbox_type, pose_type, bbox_mode, pose_mode, object_id, img, is_hidden, annotation_id, img_id, click_count, model_detecting, text_to_write, keep_processing
    img = cv2.imread(img_path)
    bbox_mode = False
    pose_mode = False
    object_id = 1
    is_detected = False
    is_hidden = 0
    img_id = None
    click_count = 0

 
    for annotation_file in ANNOTATION_FILES:

        with open(video_extraction_dir + "\\" + annotation_file, 'r') as f:
            data = json.load(f)

            for image_data in data["images"]:
                if image_data["file_name"] == img_path:
                    img_id = image_data["id"]

    if img_id == None:
        img_id = get_id(ANNOTATION_FILES, "images", video_extraction_dir)

    with open(video_extraction_dir + "/bbox_annotations.json", 'r') as f:
        data = json.load(f)

        for annotation in data["annotations"]:
            if annotation["image_id"] == img_id:
                if annotation["type"] == "detected bounding_box":
                    is_detected = True
                    break

    if is_detected == False and MODEL_FILE != "" and MODEL_FILE != None and not isinstance(MODEL_FILE, tuple) and model_detecting == "On":
        bbox_values = model.predict(img_path, conf=CONF_THRESHOLD)[0].boxes
        num_of_objects = len(bbox_values.conf)
        conf_list = []
        for i in range(num_of_objects):
            conf = bbox_values.conf[i].item()
            conf_list.append(conf)
            pred_x1, pred_y1, pred_x2, pred_y2 = map(int, bbox_values.xyxy[i].tolist())

            annotation_id = get_id(ANNOTATION_FILES, "annotations", video_extraction_dir)
         
            info = {
                "images": {
                    "id": img_id,
                    "file_name": img_path,
                    "image_height": IMAGE_HEIGHT,
                    "image_width": IMAGE_WIDTH
                },
                "annotation": {
                    "id": annotation_id,
                    "bbox": [pred_x1, pred_y1, pred_x2, pred_y2],
                    "image_id":img_id,
                    "object_id":0,
                    "iscrowd": 0,
                    "area": (pred_y2 - pred_y1) * (pred_x2 - pred_x1),
                    "type": "detected bounding_box",
                    "conf": conf,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            save_to_json(info, "bbox", video_extraction_dir, ANNOTATION_FILES)
            annotation_id += 1
        # img = cv2.imread(img_path)
        # cv2.destroyAllWindows()
        # drawing_annotations(img)
        # # text_to_write = f"Click middle of detected box with correct ID - {object_id}"
        # show_image()

      
    
        # with open(video_extraction_dir + "/bbox_annotations.json", 'r') as f:
        #     data = json.load(f)
        # keep_processing = True
        # while keep_processing and any(annotation["object_id"] == 0 for annotation in data["annotations"] if annotation["image_id"] == img_id):
        #     cv2.setMouseCallback(img_name, add_num_to_detections)
        
        #     key = cv2.waitKey(1)
        #     if key == ord('n'): # "N": Next object ID
        #         object_id += 1
          
        #         img = cv2.imread(img_path)

        #         drawing_annotations(img)
              
        #         text_to_write = f"Click middle of detected box with correct ID - {object_id}"
        #         show_image()

        #     if key == ord('d'): # "D": Delete annotations for current image
        #         object_id = 1 
               
        #         for annotation_file in ANNOTATION_FILES:
        #             with open(video_extraction_dir + annotation_file, 'r') as f:
        #                 data = json.load(f)
        #             data["annotations"] = [annotation for annotation in data["annotations"] if annotation["image_id"] != img_id]
        
        #             with open(video_extraction_dir + annotation_file, 'w') as f:
        #                 json.dump(data, f, indent=4)

        #         img = cv2.imread(img_path)
        #         text_to_write = " "
        #         show_image()
        #         cv2.setMouseCallback(img_name, dummy_function)
                
        #         break

        #     if key == 26:
        #         is_empty = True

        #         for annotation_file in ANNOTATION_FILES:
        #             with open(video_extraction_dir + annotation_file, 'r') as f:
        #                 data = json.load(f)

        #             if any(annotation["image_id"] == img_id for annotation in data["annotations"]):
        #                 is_empty = False
        #                 break
                
        #         if is_empty:
        #             handle_prev_img()
        #             object_id = 1
        #             return
        #         else:
        #             latest_time = None
                
        #         for annotation_file in ANNOTATION_FILES:
        #             with open(video_extraction_dir + annotation_file, 'r') as f:
        #                 data = json.load(f)
                    
        #             for i, annotation in enumerate(data["annotations"]):
        #                 if annotation["image_id"] == img_id:
        #                     timestamp = datetime.strptime(annotation["time"], "%Y-%m-%d %H:%M:%S")
        #                     if latest_time is None or timestamp > latest_time:
        #                         latest_time = timestamp
                


        #         for annotation_file in ANNOTATION_FILES:
        #             with open(video_extraction_dir + annotation_file, 'r') as f:
        #                 data = json.load(f)
                    
        #             for i in range(len(data["annotations"])):
        #                 timestamp = datetime.strptime(data["annotations"][i]["time"], "%Y-%m-%d %H:%M:%S")
        #                 if timestamp == latest_time:
        #                     object_id = data["annotations"][i]["object_id"]
        #                     if data["annotations"][i]["type"] == "pose":
        #                         if len(data["annotations"][i]["keypoints"]) > 1:
        #                             data["annotations"][i]["keypoints"].pop()
        #                             break
        #                         else:
                                    
        #                             del data["annotations"][i]
        #                             break
        #                     else:
            
        #                         del data["annotations"][i]
        #                     break


        #             with open(video_extraction_dir + annotation_file, 'w') as f:
        #                 json.dump(data, f, indent=4)
                
        #         img = cv2.imread(img_path)
        #         drawing_annotations(img)


        #         text_to_write = " "
        #         show_image()
        #         break
        #     if keep_processing == False:
        #         break
    #cv2.destroyAllWindows()
    breakout = False
    img = cv2.imread(img_path)
    drawing_annotations(img)
    text_to_write = None
    show_image()

    while True:
        key = cv2.waitKey(1)
        if key == 27 or cv2.getWindowProperty(img_name, cv2.WND_PROP_VISIBLE) < 1: # "Escape": Exits the program 
            cleaning(ANNOTATION_FILES, video_extraction_dir)
          

            sys.exit()
        
        elif key == ord('r'): # "R": Begins retraining the model with the annotations a user chooses
            retrain()
          
        elif key == ord('j'):
            if object_id == 0:
                object_id = 0
            else:
                object_id -= 1

            # reread the image but with a new object id and the same bbox titles as before 
            if bbox_mode == True:
                img = cv2.imread(img_path)
                drawing_annotations(img)
                if is_hidden == 1:
                    text_to_write = f"Bounding Box Mode - Hidden - {object_id}"
                elif bbox_type == "feces":
                    text_to_write = f"Bounding Box Mode - Feces"
                elif bbox_type == "normal":
                    text_to_write = f"Bounding Box Mode - {object_id}"
                
      
                show_image()

            # initialize a new pose annotation when a new object id is created 
            elif pose_mode ==  True:
            
                img = cv2.imread(img_path)
                drawing_annotations(img)

                pose_mode_text = f"Pose Mode - {object_id}"
                if pose_type:
                    pose_mode_text = f"Pose Mode - {pose_type.capitalize()} - {object_id}"
                    annotation_id = get_id(ANNOTATION_FILES, "annotations", video_extraction_dir)
                    info = {
                        "images": {
                            "id": img_id,
                            "file_name": img_path,
                            "image_height": IMAGE_HEIGHT,
                            "image_width": IMAGE_WIDTH
                        },
                        "annotation": {
                            "id": annotation_id,
                            "keypoints": [],
                            "image_id": img_id,
                            "object_id": object_id,
                            "iscrowd": 0,
                            "type": "pose",
                            "conf": 1,
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                    }
                    save_to_json(info, "pose", video_extraction_dir, ANNOTATION_FILES)

                text_to_write = pose_mode_text
                show_image()

        elif key == ord('v'): # "V": Making video of annotated images belonging to a video
    
            make_video(ANNOTATION_FILES, ANNOTATION_COLORS, IMAGE_WIDTH, IMAGE_HEIGHT)
 
        elif key == ord('m'): # "M": Turns model detection on or off, as shown on the image
            img = cv2.imread(img_path)
            model_detecting = "Off" if model_detecting == "On" else "On"
            drawing_annotations(img)
            show_image()

        elif key == ord('b'): # "B": Drawing bounding box annotations 
            if bbox_mode == False: 
                bbox_mode = True
                click_count = 0
                img = cv2.imread(img_path)
                drawing_annotations(img)
                text_to_write = f"Bounding Box Mode - {object_id}"
                show_image()
                
                cv2.setMouseCallback(img_name, drawing_bbox)  
            
                pose_mode = False
                bbox_type = "normal"

            else:
                bbox_mode = False
                bbox_type = "normal"
                is_hidden = 0
                text_to_write = None
                img = cv2.imread(img_path)
                drawing_annotations(img)
           
                show_image()
                cv2.setMouseCallback(img_name, dummy_function)
        
        elif key == ord('p'): # "P": Drawing pose annotations
            if pose_mode == False:
                pose_mode = True
                click_count = 0
                img = cv2.imread(img_path)
                drawing_annotations(img)
                text_to_write = f"Pose Mode - {object_id}"
                show_image()
            
                bbox_mode = False
                pose_type = ""
                annotation_id = get_id(ANNOTATION_FILES, "annotations", video_extraction_dir)
                
                # initializes a new pose annotation in the pose json file 
                info = {
                    "images": {
                        "id": img_id,
                        "file_name": img_path,
                        "image_height": IMAGE_HEIGHT,
                        "image_width": IMAGE_WIDTH
                    },
                    "annotation": {
                        "id": annotation_id,
                        "keypoints": [],
                        "image_id":img_id,
                        "object_id":object_id,
                        "iscrowd": 0,
                        "type": "pose",
                        "conf": 1,
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                }
                save_to_json(info, "pose", video_extraction_dir, ANNOTATION_FILES)
                cv2.setMouseCallback(img_name, dummy_function)
            else:
                text_to_write = None
                pose_mode = False
                img = cv2.imread(img_path)
                drawing_annotations(img)
                
                show_image()
                cv2.setMouseCallback(img_name, dummy_function)


        elif key == 13: # "Enter": Go to the next image
            #cv2.destroyAllWindows()
            pose_mode = False
            bbox_mode = False
            already_passed = False
            object_id = 1
            show_image()
            return

        elif key == 8: # "Backspace": Go back to previous image  
            show_image()
           
            handle_prev_img()
            
            return

        elif key == ord('d'): # "D": Delete all annotations for the current image
            for annotation_file in ANNOTATION_FILES:
                with open(video_extraction_dir + "\\" + annotation_file, 'r') as f:
                    data = json.load(f)
                data["annotations"] = [annotation for annotation in data["annotations"] if annotation["image_id"] != img_id]
    
                with open(video_extraction_dir + "\\" + annotation_file, 'w') as f:
                    json.dump(data, f, indent=4)

            img = cv2.imread(img_path)
            text_to_write = None
            show_image()
            cv2.setMouseCallback(img_name, dummy_function)
            bbox_mode = False
            pose_mode = False
            object_id = 1
            continue

        elif key == 26: # "Ctrl + Z": Undo most recent annotation, if are none annotations go to previous image 
            is_empty = True

            for annotation_file in ANNOTATION_FILES:
                with open(video_extraction_dir + "\\" + annotation_file, 'r') as f:
                    data = json.load(f)

                if any(annotation["image_id"] == img_id for annotation in data["annotations"]):
                    is_empty = False
                    break
            
            if is_empty:
                handle_prev_img()
                object_id = 1
                return
            else:
                latest_time = None
            
            # finding the latest timestamp for an annotation for the current image id 
            for annotation_file in ANNOTATION_FILES:
                with open(video_extraction_dir + "\\" + annotation_file, 'r') as f:
                    data = json.load(f)
                
                for i, annotation in enumerate(data["annotations"]):
                    if annotation["image_id"] == img_id:
                        timestamp = datetime.strptime(annotation["time"], "%Y-%m-%d %H:%M:%S")
                        if latest_time is None or timestamp > latest_time:
                            latest_time = timestamp
            

            # deleting the annotation with the latest timestamp for an annotation for the current image id 
            for annotation_file in ANNOTATION_FILES:
                with open(video_extraction_dir + "\\" + annotation_file, 'r') as f:
                    data = json.load(f)
                
                for i in range(len(data["annotations"])):
                    timestamp = datetime.strptime(data["annotations"][i]["time"], "%Y-%m-%d %H:%M:%S")
                    if timestamp == latest_time:
                        object_id = data["annotations"][i]["object_id"]
                        if data["annotations"][i]["type"] == "pose":
                            if len(data["annotations"][i]["keypoints"]) > 1:
                                data["annotations"][i]["keypoints"].pop()
                                break
                            else:
                                
                                del data["annotations"][i]
                                break
                        else:
        
                            del data["annotations"][i]
                        break


                with open(video_extraction_dir + "\\" + annotation_file, 'w') as f:
                    json.dump(data, f, indent=4)
            
            img = cv2.imread(img_path)
            drawing_annotations(img)

            # rewriting the previous titles after deletion
            mode_text = ""
            if bbox_mode:
                mode_text = "Bounding Box Mode - "
                if is_hidden:
                    mode_text += "Hidden - "
                elif bbox_type == "feces":
                    mode_text += "Feces - "
                mode_text += str(object_id)
            elif pose_mode:
                mode_text = "Pose Mode - "
                if pose_type:
                    mode_text += f"{pose_type.capitalize()} - "
                mode_text += str(object_id)

            
            already_passed = False
            drawing_annotations(img)
            text_to_write = mode_text
            show_image()
            
        elif key == ord('n'): # "N": Next object ID
            object_id += 1

            # reread the image but with a new object id and the same bbox titles as before 
            if bbox_mode == True:
                img = cv2.imread(img_path)
                drawing_annotations(img)
                if is_hidden == 1:
                    text_to_write = f"Bounding Box Mode - Hidden - {object_id}"
                elif bbox_type == "feces":
                    text_to_write = f"Bounding Box Mode - Feces"
                elif bbox_type == "normal":
                    text_to_write = f"Bounding Box Mode - {object_id}"
                
      
                show_image()

            # initialize a new pose annotation when a new object id is created 
            if pose_mode ==  True:
            
                img = cv2.imread(img_path)
                drawing_annotations(img)

                pose_mode_text = f"Pose Mode - {object_id}"
                if pose_type:
                    pose_mode_text = f"Pose Mode - {pose_type.capitalize()} - {object_id}"
                    annotation_id = get_id(ANNOTATION_FILES, "annotations", video_extraction_dir)
                    info = {
                        "images": {
                            "id": img_id,
                            "file_name": img_path,
                            "image_height": IMAGE_HEIGHT,
                            "image_width": IMAGE_WIDTH
                        },
                        "annotation": {
                            "id": annotation_id,
                            "keypoints": [],
                            "image_id": img_id,
                            "object_id": object_id,
                            "iscrowd": 0,
                            "type": "pose",
                            "conf": 1,
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                    }
                    save_to_json(info, "pose", video_extraction_dir, ANNOTATION_FILES)

                text_to_write = pose_mode_text
                show_image()
            

        # when in bbox mode, user can select 'f' for a bbox for feces or 'h' for a bbox that is partly hidden
        if bbox_mode:
        
            bbox_options = {
                ord('f'): ("feces", "Feces"),
                ord('h'): ("normal", "Hidden")
            }

            for keybind, (bbox_label, mode_message) in bbox_options.items():
                if key == keybind:
                    img = cv2.imread(img_path)
                    drawing_annotations(img)
                    text_to_write = f"Bounding Box Mode - {mode_message} - {object_id}"
                    
                    show_image()
                
                    is_hidden = 1 if bbox_label == "normal" else 0
                    bbox_type = bbox_label.lower()
                    cv2.setMouseCallback(img_name, drawing_bbox)
                

        # when in pose mode, user can select '1', '2', or '3' for different parts of the object
        elif pose_mode:

            pose_options = {
            ord('1'): ("Head"),
            ord('2'): ("Tail"),
            ord('3'): ("Neck")
        }

            for keybind, p_label in pose_options.items():
                if key == keybind:
                    img = cv2.imread(img_path)
                    drawing_annotations(img)
                    text_to_write =   f"Pose Mode - {p_label} - {object_id}"
                    
                    show_image()
                    pose_type = p_label.lower()
                    cv2.setMouseCallback(img_name, drawing_pose)
                    
    



if __name__ == "__main__":
    # initializing constants 
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_skip', type=int, default=50, help='Number of frames to skip')
    args = parser.parse_args()
    FRAME_SKIP = args.frame_skip

    ANNOTATION_FILES = ["bbox_annotations.json", "pose_annotations.json"]
    FONT_SCALE = 0.5
    FONT_THICKNESS = 1
    FONT_COLOR = (255, 255, 0)
    ANNOTATION_COLORS = []
    CONF_THRESHOLD = 0.25
    
    screen = screeninfo.get_monitors()[0]  # Assuming you want the primary monitor
    width, height = screen.width, screen.height
    screen_center_x = int((width - 700) / 2)
    screen_center_y = int((height - 500)/ 2)
    # creating a list of random annotation colors that are- the same throughout different runs 
    seed = 42
    random.seed(seed)
    for _ in range(30):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)

        color = (r, g, b)
        ANNOTATION_COLORS.append(color)

    coordinate_stack = {"img_name": None, "coordinates": None, "dimensions": None}
    # to get the sizing for putting text at appropiate places on the cv2 window
    textSize, baseline = cv2.getTextSize("test", cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)
    textSizeWidth, textSizeHeight = textSize


    # user will choose the video file and/or model file, video file will be extracted as frames into a image directory
    MODEL_FILE, video_name, video_extraction_dir = extract_frames(FRAME_SKIP)
    IMAGE_DIR = "used_videos/" + video_name.split(".")[0] + "/extracted_frames/"

    assert IMAGE_DIR is not None, "A image folder was empty."

    # initialize the json files in the respective video directory
    for annotation_file in ANNOTATION_FILES:
        print(os.path.exists(video_extraction_dir + "\\" + annotation_file))
        if not os.path.exists(video_extraction_dir + "\\" + annotation_file):
            json_content = {"images": [], "annotations": []}
            
            with open(video_extraction_dir + "\\" + annotation_file, 'w') as f:
                json.dump(json_content, f, indent=4)
    DIR_LIST = None

    # if a model is selected, otherwise let the user annotate with no model assistance
    if not isinstance(MODEL_FILE, tuple) and MODEL_FILE != "" and MODEL_FILE != None:
        print("CUDA available?: ", torch.cuda.is_available())
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = YOLO(MODEL_FILE)
        model.to(device)

        # comment this to turn off clustering 
        # clustering_data(IMAGE_DIR, MODEL_FILE)
        # DIR_LIST = os.listdir("used_videos/" + video_name.split(".")[0] + "/clusters/")
        # for i, dir in enumerate(DIR_LIST):
        #     DIR_LIST[i] = "used_videos/" + video_name.split(".")[0] + "/clusters/" + dir + "/" 
        #shutil.rmtree(IMAGE_DIR, ignore_errors=True)
    else:
        DIR_LIST = None
        
    already_passed = False
    object_id = 1
    annotations_exists = False
    model_detecting = "On"

    for annotation_file in ANNOTATION_FILES:
        with open(video_extraction_dir + "\\" + annotation_file, 'r') as f:
            data = json.load(f)

        if data["annotations"]:
            annotations_exists = True
            break


    # Create a window pop up to determine if user wants to continue where they left off 
    if annotations_exists:
        window = Tk()
        window.attributes('-topmost', True)
        window.withdraw()
        result = messagebox.askquestion("Continue to Next Image", "Do you want to continue your work on the image following the last annotated image?")

        window.destroy()

    img_num = 0
  
    # directory list will be the list of clusters if a model is chosen, or a list of extracted frames
    directories = [IMAGE_DIR] if not DIR_LIST else DIR_LIST

    for current_dir in directories:
        
        imgs = os.listdir(current_dir)
        text_to_write = None
        img_path = os.path.join(current_dir, imgs[0])
        img_name = os.path.basename(img_path)
        img_name = img_path
        img = cv2.imread(img_path)
        IMAGE_HEIGHT, IMAGE_WIDTH = img.shape[:2]

        # print("Initializing images...")
        # for img_n in range(len(imgs)):
        #     img_path = os.path.join(current_dir, imgs[img_n])
        #     img_name = os.path.basename(img_path)
        #     img = cv2.imread(img_path)
        #     show_image()
        #     # win = gw.getWindowsWithTitle(img_name)[0]
        #     # win.minimize()


        #     cv2.destroyAllWindows()
   
      
        print("Completed")
        while img_num < len(imgs):
            is_hidden = 0
            annotations_exists = False
            annotated_image_ids = set()
            img_num = int(img_num)
            img_path = os.path.join(current_dir, imgs[img_num])
            img_name = os.path.basename(img_path)
   
            if int(((img_name.split('_'))[-1]).replace('.jpg', '')) % FRAME_SKIP == 0:

                img = cv2.imread(img_path)

                IMAGE_HEIGHT, IMAGE_WIDTH = img.shape[:2]

                annotated_image_ids = cleaning(ANNOTATION_FILES, video_extraction_dir)
                if annotated_image_ids and already_passed == False:
                    for annotation_file in ANNOTATION_FILES:
                        with open(os.path.join(video_extraction_dir, annotation_file), 'r') as f:
                            data = json.load(f)

                        if len(data["images"]) == 0:
                            break

                        for image_data in data["images"]:
                            if image_data["file_name"] == img_path:
                                if image_data["id"] in annotated_image_ids:
                                    annotations_exists = True
                                    break

                if not annotations_exists:
                    annotating(img_path, img_name, video_extraction_dir)
                else:
                    if result == "no":
                        annotating(img_path, img_name, video_extraction_dir)

            img_num += 1

        img_num = 0  # Reset img_num for the next directory

    cv2.destroyAllWindows()
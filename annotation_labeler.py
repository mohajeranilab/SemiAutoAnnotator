import cv2
import os
from pathlib import Path
import torch
from ultralytics import YOLO 
import numpy as np
import random
import json
from datetime import datetime
from tkinter import Tk, filedialog, messagebox
import shutil
import sys

# CTRL ZING NOT WORKING

def dummy_function(event, x, y, flags, param):
    pass


def extract_frames():
    """
    Extracts frames from a selected video file at a specified interval (FRAME_SKIP) 
    and saves them to used_videos/{video_file_name}/extracted_frames/

    After, it prompts the user to select a model/weights file 
    It also copies the video file to the used_videos/{video_file_name}/ directory

    """


    root = Tk()
    # Set the window to be always on top
    root.attributes('-topmost', True)
    # Hide the root window
    root.withdraw()

    # File explorer pop up for selecting which video file to use
    video_path = filedialog.askopenfilename(
        initialdir="/", 
        title="SELECT VIDEO FILE",
        filetypes=(("Video files", "*.mp4;*.avi;*.mov;*.mkv"), ("All files", "*.*"))
    )

    if not video_path:
        print("No video file selected.")
        sys.exit()

    video_name, _ = os.path.splitext(os.path.basename(video_path))
    video_extraction_dir = os.path.join("used_videos", video_name)
    
    # Create a folder using the video name with further subdirectories to be added
    os.makedirs(video_extraction_dir, exist_ok=True)
    # Copy the video to the folder
    shutil.copy(video_path, "used_videos/" + video_name.split(".")[0])

    if not os.path.exists(video_extraction_dir + "/extracted_frames/"):
        os.makedirs(video_extraction_dir + "/extracted_frames/")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            sys.exit()
        
        frame_count = 0
        extracted_count = 0

        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

            ret, frame = cap.read()
            
            if not ret:
                break

            # Skipping by FRAME_SKIP to shorten how many frames are needed to annotate
            if frame_count % FRAME_SKIP == 0:
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                frame_filename = os.path.join(video_extraction_dir + "/extracted_frames/", f"{base_name}_img_{frame_count:05d}.jpg")
             
                cv2.imwrite(frame_filename, frame)
                extracted_count += 1
            
            frame_count += FRAME_SKIP 

        cap.release()
        print(f"Extracted {extracted_count} frames to 'extracted_frames'")

    else:
        print(f"Frames already exist at used_videos/{video_name.split('.')[0]}/extracted_frames/. \nIf you want to change how many frames are extracted in the video, make sure to delete the used_videos/{video_name.split('.')[0]}/extracted_frames/ directory")
    
    # File explorer pop up for selecting which model to use
    model_path = filedialog.askopenfilename(initialdir="/", title="SELECT MODEL/WEIGHTS FILE")
    
    root.destroy()
    return model_path, video_name, video_extraction_dir

def clustering_data():
    pass


def save_annotations_to_json(annotations, type):
    """
    Saves annotations made by the user, bbox or pose, to their respective json file

    annotations (dict): Created annotation to be saved into the respective json file
    type (str): The type of annotation, "bbox" or "pose"

    """ 

    file_index = {
        "bbox": 0,
        "pose": 1
    }.get(type, None)

    if file_index is None:
        return  

        
    with open(video_extraction_dir + ANNOTATION_FILES[file_index], 'r') as f:
        data = json.load(f)

    image_id = annotations["images"]["id"]
    if any(image["id"] == image_id for image in data["images"]):
        pass
    else:
        data["images"].append(annotations["images"])

    data["annotations"].append(annotations["annotation"])
    with open(video_extraction_dir + ANNOTATION_FILES[file_index], 'w') as f:
        json.dump(data, f, indent=4)

def make_video():

    video_path = filedialog.askdirectory(
        initialdir="/", 
        title="SELECT VIDEO FOLDER IN used_videos/",
        
    )
    print(video_path)

    print("Combining annotated frames to video ......")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter("output_video.avi", fourcc, 30.0, (IMAGE_WIDTH, IMAGE_HEIGHT))
    frames_to_write = {}

    # First, gather all frames to write
    for annotation_file in ANNOTATION_FILES:
     
        with open(video_path + '/' + annotation_file, 'r') as f:
            data = json.load(f)
        if len(data["annotations"]) != 0:
            for image_data in data["images"]:
                frames_to_write[image_data["id"]] = image_data["file_name"]
        else:
            continue
    if not frames_to_write:
        print("No annotations have been made for the selected video")


    all_annotations = {}

    for annotation_file in ANNOTATION_FILES:
        with open(video_path + '/' + annotation_file, 'r') as f:
            data = json.load(f)
        
        for annotation in data["annotations"]:
            image_id = annotation["image_id"]
            if image_id not in all_annotations:
                all_annotations[image_id] = []
            all_annotations[image_id].append((annotation_file, annotation))

    # Draw all annotations on each frame
    for image_id, image_file in frames_to_write.items():
        if image_id in all_annotations:
            image = cv2.imread(image_file)
            
            for annotation_file, annotation in all_annotations[image_id]:
                if annotation_file == "/bbox_annotations.json":
                    image = cv2.rectangle(image, (annotation["bbox"][0], annotation["bbox"][1]), 
                                        (annotation["bbox"][2], annotation["bbox"][3]), 
                                        ANNOTATION_COLORS[annotation["object_id"]], 2)

                elif annotation_file == "/pose_annotations.json":
                    for keypoint_annotation in annotation["keypoints"]:
                        if len(keypoint_annotation) != 0:
                            image = cv2.circle(image, (keypoint_annotation[1][0], keypoint_annotation[1][1]), 
                                            5, ANNOTATION_COLORS[annotation["object_id"]], -1)

            video.write(image)

    video.release()

    print("Video has been created called output_video")
    




def show_image(): 
    """
    Shows the image, also resizes it to a specific size and also moves it to a specific place on the screen

    Moving to a specific place solved my issue of the cv2 window opening in random parts of the screen 
    """

    cv2.namedWindow(img_name, cv2.WINDOW_NORMAL)  
    cv2.resizeWindow(img_name, 700, 500)  
    
    cv2.moveWindow(img_name, 900, 320)
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

    if img_num > 0:
        img_num -= 2 
        cv2.destroyAllWindows()
        return
    elif img_num == 0:
        img_num -= 1
        cv2.destroyAllWindows()
        return 

def get_id(annotation_files, data_type):
    """
    annotation_files (list): a list of the annotation files
    data_type (str): "annotations" or "images", whether to look at the image or annotation id.

    Returns:
    id (int): the unique id of image or annotation
    """

    id_set = set()
    for annotation_file in annotation_files:
        with open(video_extraction_dir + annotation_file, 'r') as f:
            data_file = json.load(f)
        if len(data["images"]) == 0:
            id = 0
        id_set.update(data["id"] for data in data_file[data_type])
    id = 0
    while id in id_set:
        id += 1
    return id



def drawing_annotations(img):
    """
    This will draw the existing annotations on the cv2 image

    img (numpy.ndarray): cv2 image
    """
    annotation_types = ["bbox", "pose"]

    for type_idx, annotation_type in enumerate(annotation_types):
        with open(video_extraction_dir + ANNOTATION_FILES[type_idx], 'r') as f:
            annotations = json.load(f)
        
        for annotation in annotations["annotations"]:
            if annotation["image_id"] == img_id:
                if annotation_type == "bbox":
                    img = cv2.rectangle(img, (annotation["bbox"][0], annotation["bbox"][1]), (annotation["bbox"][2], annotation["bbox"][3]), ANNOTATION_COLORS[annotation["object_id"]], 2)
                    if annotation["type"] == "detected_bbox":
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
    annotation_id = get_id(ANNOTATION_FILES, "annotations")
    img_id = None
    for annotation_file in ANNOTATION_FILES:
        with open(video_extraction_dir + annotation_file, 'r') as f:
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
        img_id = get_id(ANNOTATION_FILES, "images")
  

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
            save_annotations_to_json(info, "bbox")
   
    
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
    video_files = []
    print("""\nSelect video files you want to use to train. These video files must have been annotated previously and annotations saved in their directory.\n
Press Cancel or exit out of the file explorer when finished choosing videos.
    """)
    while True:
        video_file = filedialog.askopenfilename(
            title="SELECT VIDEO FILES TO TRAIN",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if video_file == "":
            break
  
        if video_file not in video_files:
            video_files.append(video_file)
        
    root.destroy()
    if "" in video_files:
        video_files.remove('')
    if len(video_files) == 0:
        print("There are no video files selected")
        sys.exit()


    # Creating the textfile with annotations to its respective directory
    for video_file in video_files:
   
        if not os.path.exists("used_videos/" + str(os.path.splitext(os.path.basename(video_file))[0]) + "/extracted_frames"):

            print(f"There are no annotations made for {video_file}")
            sys.exit()
        else:
            if not os.path.exists("used_videos/" + str(os.path.splitext(os.path.basename(video_file))[0]) + "/bbox_annotations.json"):
                print(f"There are no annotations made for {video_file}")
                sys.exit()

            else:
                with open("used_videos/" + str(os.path.splitext(os.path.basename(video_file))[0]) + "/bbox_annotations.json", 'r') as f:
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
                                x_center = (annotation["bbox"][0]/IMAGE_WIDTH) + width 
                                y_center = (annotation["bbox"][1]/IMAGE_HEIGHT) + height 

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

def annotating(img_path, img_name, video_extraction_dir):
    global already_passed, bbox_type, pose_type, bbox_mode, pose_mode, object_id, img, is_hidden, annotation_id, img_id, click_count, model_detecting, text_to_write, img_num
    img = cv2.imread(img_path)
    bbox_mode = False
    pose_mode = False
    object_id = 1
    is_detected = False
    is_hidden = 0
    img_id = None
    click_count = 0
    
   
 
    for annotation_file in ANNOTATION_FILES:
        with open(video_extraction_dir + annotation_file, 'r') as f:
            data = json.load(f)

            for image_data in data["images"]:
                if image_data["file_name"] == img_path:
                    img_id = image_data["id"]

    if img_id == None:
        img_id = get_id(ANNOTATION_FILES, "images")

    with open(video_extraction_dir + "/bbox_annotations.json", 'r') as f:
        data = json.load(f)

        for annotation in data["annotations"]:
            if annotation["image_id"] == img_id:
                if annotation["type"] == "detected_bbox":
                    is_detected = True
                    break

    if is_detected == False and model_detecting == "On":
        bbox_values = model.predict(img_path, conf=CONF_THRESHOLD)[0].boxes
        num_of_objects = len(bbox_values.conf)
        conf_list = []
        for i in range(num_of_objects):
            conf = bbox_values.conf[i].item()
            conf_list.append(conf)
            pred_x1, pred_y1, pred_x2, pred_y2 = map(int, bbox_values.xyxy[i].tolist())

            annotation_id = get_id(ANNOTATION_FILES, "annotations")
         
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
                    "object_id":object_id,
                    "iscrowd": 0,
                    "area": (pred_y2 - pred_y1) * (pred_x2 - pred_x1),
                    "type": "detected_bbox",
                    "conf": conf,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            save_annotations_to_json(info, "bbox")
            annotation_id += 1
        
    breakout = False
    drawing_annotations(img)
    text_to_write = " "
    show_image()

    while True:
        key = cv2.waitKey(1)
        if key == 27 or cv2.getWindowProperty(img_name, cv2.WND_PROP_VISIBLE) < 1: # "Escape": Exits the program 
            cleaning(video_extraction_dir)
          

            sys.exit()
        
        elif key == ord('r'): # "R": Begins retraining the model with the annotations a user chooses
            retrain()
          

        elif key == ord('v'): # "V": Making video of annotated images belonging to a video
    
            make_video()

        elif key == ord('m'): 
            img = cv2.imread(img_path)
            model_detecting = "Off" if model_detecting == "On" else "On"
            drawing_annotations(img)

            show_image()
            img_num -= 1
            return
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
                annotation_id = get_id(ANNOTATION_FILES, "annotations")
                
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
                save_annotations_to_json(info, "pose")
                cv2.setMouseCallback(img_name, dummy_function)
            else:
                pose_mode = False
                img = cv2.imread(img_path)
                drawing_annotations(img)
                
                show_image()
                cv2.setMouseCallback(img_name, dummy_function)


        elif key == 13: # "Enter": Go to the next image
            cv2.destroyAllWindows()
            pose_mode = False
            bbox_mode = False
            already_passed = False
            object_id = 1
            return

        elif key == 8: # "Backspace": Go back to previous image  
            handle_prev_img()
            return

        elif key == ord('d'): # "D": Delete all annotations for the current image
            for annotation_file in ANNOTATION_FILES:
                with open(video_extraction_dir + annotation_file, 'r') as f:
                    data = json.load(f)
                data["annotations"] = [annotation for annotation in data["annotations"] if annotation["image_id"] != img_id]
    
                with open(video_extraction_dir + annotation_file, 'w') as f:
                    json.dump(data, f, indent=4)

            img = cv2.imread(img_path)
        
            show_image()
            # cv2.setMouseCallback(img_name, dummy_function)
            # bbox_mode = False
            # pose_mode = False
            # object_id = 1
            continue

        elif key == 26: # "Ctrl + Z": Undo most recent annotation, if are none annotations go to previous image 
            is_empty = True

            for annotation_file in ANNOTATION_FILES:
                with open(video_extraction_dir + annotation_file, 'r') as f:
                    data = json.load(f)
                
                if any(annotation["image_id"] == img_id for annotation in data["annotations"]):
                    is_empty = False
                    break
            
            if is_empty:
                handle_prev_img()
                object_id = 1
                return

            latest_time = max(datetime.strptime(annotation["time"], "%Y-%m-%d %H:%M:%S") for annotation in data["annotations"] if annotation["image_id"] == img_id)
            
            for annotation_file in ANNOTATION_FILES:
                with open(video_extraction_dir + annotation_file, 'r') as f:
                    data = json.load(f)
                
                for annotation in data["annotations"]:
                    if annotation["time"] == latest_time.strftime("%Y-%m-%d %H:%M:%S"):
                        object_id = annotation["object_id"]
                        if annotation["type"] == "pose":
                            if annotation["keypoints"]:
                                annotation["keypoints"].pop()
                            else:
                                data["annotations"].remove(annotation)
                        else:
                            data["annotations"].remove(annotation)
                        break
                
                with open(video_extraction_dir + annotation_file, 'w') as f:
                    json.dump(data, f, indent=4)
            
            img = cv2.imread(img_path)
            drawing_annotations(img)

            mode_text = ""
            if bbox_mode:
                mode_text = "Bounding Box Mode - "
                if is_hidden:
                    mode_text += "Hidden - "
                elif bbox_type == "feces":
                    mode_text += "Feces - "
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

            if pose_mode ==  True:
            
                img = cv2.imread(img_path)
                drawing_annotations(img)

                pose_mode_text = f"Pose Mode - {object_id}"
                if pose_type:
                    pose_mode_text = f"Pose Mode - {pose_type.capitalize()} - {object_id}"
                    annotation_id = get_id(ANNOTATION_FILES, "annotations")
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
                    save_annotations_to_json(info, "pose")

                text_to_write = pose_mode_text
                show_image()
            

        
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
                    
    
def cleaning(video_extraction_dir):
    for annotation_file in ANNOTATION_FILES:
        with open(video_extraction_dir + annotation_file, 'r') as f:
            data = json.load(f)

        if len(data["images"]) == 0:
            continue

        annotated_image_ids = {annotation["image_id"] for annotation in data["annotations"]}

        


        data["images"] = [image_data for image_data in data["images"] if image_data["id"] in annotated_image_ids]
        if annotation_file == "/pose_annotations.json":
            data["annotations"] = [annotation_data for annotation_data in data["annotations"] if annotation_data["keypoints"]]
        
        with open(video_extraction_dir + annotation_file, 'w') as f:
            json.dump(data, f, indent=4)
        
        return annotated_image_ids


if __name__ == "__main__":

    ANNOTATION_FILES = ["/bbox_annotations.json", "/pose_annotations.json"]
    FONT_SCALE = 0.5
    FONT_THICKNESS = 1
    FONT_COLOR = (255, 255, 0)
    ANNOTATION_COLORS = []
    seed = 42
    random.seed(seed)

    for _ in range(30):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)

        color = (r, g, b)
        ANNOTATION_COLORS.append(color)

    CONF_THRESHOLD = 0.25
    FRAME_SKIP = 100


    textSize, baseline = cv2.getTextSize("test", cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)
    textSizeWidth, textSizeHeight = textSize


 
 
    MODEL_DIR, video_name, video_extraction_dir = extract_frames()
    IMAGE_DIR = "used_videos/" + video_name.split(".")[0] + "/extracted_frames/"


    if MODEL_DIR == None or IMAGE_DIR == None:
        print("A selected file/folder was empty.")
        exit()

    for annotation_file in ANNOTATION_FILES:
        if not os.path.exists(video_extraction_dir + annotation_file):
            json_content = {"images": [], "annotations": []}
            
            with open(video_extraction_dir + annotation_file, 'w') as f:
                json.dump(json_content, f, indent=4)

    print("CUDA available?: ", torch.cuda.is_available())
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = YOLO(MODEL_DIR)
    model.to(device)


    img_num = 0
    imgs = os.listdir(IMAGE_DIR)
    already_passed = False
    object_id = 1
    annotations_exists = False
    model_detecting = "On"
    for annotation_file in ANNOTATION_FILES:
        with open(video_extraction_dir + annotation_file, 'r') as f:
            data = json.load(f)

        if data["annotations"]:
            annotations_exists = True
            break

    # Only execute the Tkinter code if annotations exist
    if annotations_exists:
        window = Tk()
        window.attributes('-topmost', True)
        window.withdraw()
        result = messagebox.askquestion("Continue to next image?", "Do you want to continue your work on the image following the last annotated image?")
        window.destroy()
    

    while img_num < len(imgs):
        
        is_hidden = 0
        annotations_exists = False
        annotated_image_ids = set()
        img_num = int(img_num)
        img_path = os.path.join(IMAGE_DIR, imgs[int(img_num)])
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)

        IMAGE_HEIGHT, IMAGE_WIDTH = img.shape[:2]
      

        annotated_image_ids = cleaning(video_extraction_dir)

        if already_passed == False:
            for annotation_file in ANNOTATION_FILES:
                with open(video_extraction_dir + annotation_file, 'r') as f:
                    data = json.load(f)
                
                if len(data["images"]) == 0:
                    break

                for image_data in data["images"]:
                    if image_data["file_name"] == img_path:
                       
                        if image_data["id"] in annotated_image_ids:
                            annotations_exists = True
                            break
        
        if annotations_exists == False:
            if result == "yes":
                img_num -= 1
                img_path = os.path.join(IMAGE_DIR, imgs[int(img_num)])
                img_name = os.path.basename(img_path)
                annotating(img_path, img_name, video_extraction_dir)
            else:
                annotating(img_path, img_name, video_extraction_dir)
            
        else:
            if result == "no":
                annotating(img_path, img_name, video_extraction_dir)
            else:
                pass

        
        img_num += 1

    cv2.destroyAllWindows()
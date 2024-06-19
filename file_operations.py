from tkinter import Tk, filedialog, messagebox
import cv2
import shutil
import os
import sys
import json

def extract_frames(frame_skip):
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
    # if not os.path.exists(video_extraction_dir + "/clusters/"):
        os.makedirs(video_extraction_dir + "/extracted_frames/")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        sys.exit()
    
    frame_count = 0
    extracted_count = 0

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        frame_filename = os.path.join(video_extraction_dir + "/extracted_frames/", f"{base_name}_img_{frame_count:05d}.jpg")

        if not os.path.exists(frame_filename):
            ret, frame = cap.read()
            
            if not ret:
                break
        
        # Skipping by FRAME_SKIP to shorten how many frames are needed to annotate
            if frame_count % frame_skip == 0:
                
             
                cv2.imwrite(frame_filename, frame) 
                extracted_count += 1
         
            
        frame_count += frame_skip 

    cap.release()
    print(f"Extracted {extracted_count} frames to 'extracted_frames'")

    # else:
    #     print(f"Frames already exist at used_videos/{video_name.split('.')[0]}/extracted_frames/. \nIf you want to change how many frames are extracted in the video, make sure to delete the used_videos/{video_name.split('.')[0]}/extracted_frames/ directory")
    
    # File explorer pop up for selecting which model to use
    model_path = filedialog.askopenfilename(initialdir="/", title="SELECT MODEL/WEIGHTS FILE")
    
    root.destroy()
    return model_path, video_name, video_extraction_dir


def save_to_json(annotations, type, video_extraction_dir, annotation_files):
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

        
    with open(video_extraction_dir + annotation_files[file_index], 'r') as f:
        data = json.load(f)

    image_id = annotations["images"]["id"]
    if any(image["id"] == image_id for image in data["images"]):
        pass
    else:
        data["images"].append(annotations["images"])

    data["annotations"].append(annotations["annotation"])
    with open(video_extraction_dir + annotation_files[file_index], 'w') as f:
        json.dump(data, f, indent=4)

def make_video(annotation_files, annotation_colors, image_width, image_height):

    video_path = filedialog.askdirectory(
        initialdir="/", 
        title="SELECT VIDEO FOLDER IN used_videos/",
        
    )
    print(video_path)

    print("Combining annotated frames to video ......")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter("output_video.avi", fourcc, 30.0, (image_width, image_height))
    frames_to_write = {}

    # First, gather all frames to write
    for annotation_file in annotation_files:
     
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

    for annotation_file in annotation_files:
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
                                        annotation_colors[annotation["object_id"]], 2)

                elif annotation_file == "/pose_annotations.json":
                    for keypoint_annotation in annotation["keypoints"]:
                        if len(keypoint_annotation) != 0:
                            image = cv2.circle(image, (keypoint_annotation[1][0], keypoint_annotation[1][1]), 
                                            5, annotation_colors[annotation["object_id"]], -1)

            video.write(image)

    video.release()

    print("Video has been created called output_video")
    
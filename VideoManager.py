import cv2
import os
import shutil
import json
from PyQt5.QtWidgets import QApplication, QFileDialog
from PyQt5.QtCore import Qt

class VideoManager():
    """
    The VideoManager class manages video-related operations for an annotation labeler, including extracting frames from videos and creating annotated videos.
    """

    def __init__(self, frame_skip, annotation_colors, annotation_files):
   
        self.processed_path = None
        self.frame_skip = frame_skip
        self.cv2_img = None
        self.annotation_colors = annotation_colors 
        self.annotation_files = annotation_files 


    def extract_frames(self):
        """
        Extract frames from a selected video that the user wants to annotate 

        Returns:
            video_name (str): name of selected video        
        """

      
        video_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select Video File",
            "/",
            "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*)"
        )


        assert video_path, "No video file selected."

        
        # extracting video name and creating a directory for the video
        video_name, _ = os.path.splitext(os.path.basename(video_path))
        self.processed_path = os.path.join("used_folders", video_name)
        os.makedirs(self.processed_path, exist_ok=True)


        # copying video to the 'used_folders/{video_name}/ directory
        shutil.copy(video_path, "used_folders/" + video_name.split(".")[0])
    
 
        if not os.path.exists(os.path.join(self.processed_path, "extracted_frames")):
            os.makedirs(os.path.join(self.processed_path, "extracted_frames"))

        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), "Error: Could not open video."
       
            
        frame_count = 0
        extracted_count = 0

        base_name = os.path.splitext(os.path.basename(video_path))[0]

        # extract frames
        while True:
          
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            frame_filename = os.path.join(self.processed_path, "extracted_frames", f"{base_name}_img_{frame_count:05d}.jpg")

            if not os.path.exists(frame_filename):
                ret, frame = cap.read()
                
                if not ret:
                    break
            
                # skipping by frame_skip to shorten how many frames are needed to annotate
                if frame_count % self.frame_skip == 0:
                    
                
                    cv2.imwrite(frame_filename, frame) 
                    extracted_count += 1
            
                
            frame_count += self.frame_skip 


        cap.release()
        print(f"Extracted {extracted_count} new frames to 'extracted_frames'")

        
        return video_name
    

    def image_folder_extraction(self):
        """
        Extract images from a selected folder for annotation.

        Returns:
            folder_name (str): Name of the selected folder.
        """
        folder_path = QFileDialog.getExistingDirectory(
            None,
            "Select Folder",
            "/",
            QFileDialog.ShowDirsOnly
        )

        assert folder_path, "No folder selected."

        # extracting folder name and creating a directory for the folder
        folder_name = os.path.basename(folder_path)
   
        self.folder_dir = os.path.join("used_folders", folder_name)
    
        os.makedirs(self.folder_dir, exist_ok=True)
        # creating extracted_images in the chosen folder
        if not os.path.exists(os.path.join(self.folder_dir, "extracted_frames")):
            os.makedirs(os.path.join(self.folder_dir, "extracted_frames"))
        
        # Copying folder to the 'used_folders/{folder_name}/' directory
        shutil.copytree(folder_path, os.path.join(self.folder_dir, "extracted_frames"), dirs_exist_ok=True)

        # Iterate through the image files in the folder
        extracted_count = 0

        for root, _, files in os.walk(self.folder_dir):
            for file in files:
                file_path = os.path.join(root, file)

                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    image = cv2.imread(file_path)

                    if image is not None:
                        base_name = os.path.splitext(file)[0]
                        frame_filename = os.path.join(self.folder_dir, "extracted_frames", f"{base_name}.jpg")

                        # Save the image in the 'extracted_frames' directory
                        cv2.imwrite(frame_filename, image)
                        extracted_count += 1

        print(f"Extracted {extracted_count} images to 'extracted_frames'")
        self.processed_path = self.folder_dir
        return folder_name


    def make_video(self):
        """
        Creates a video from the selected frames that have been annotated. An output video will be created 
        """

        video_path = QFileDialog.getExistingDirectory(
            None,
            "Select Folder",
            "used_folders/",
            options=QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks | QFileDialog.Option(Qt.WindowStaysOnTopHint)
        )
  

        print("Combining annotated frames to video ......")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(os.path.join(video_path, "output_video.mp4"), fourcc, 30.0, (self.cv2_img.width, self.cv2_img.height))
        frames_to_write = {}

        # read annotations from files
        for annotation_file in self.annotation_files:
       
            with open(os.path.join(video_path, annotation_file) ,'r') as f:
         
                data = json.load(f)
            if len(data["annotations"]) != 0:
                for image_data in data["images"]:
                    frames_to_write[image_data["id"]] = image_data["file_name"]

        if not frames_to_write:
            print("No annotations have been made for the selected video")

        
        # collecting all annotations for each frame
        all_annotations = {}
        
        for annotation_file in self.annotation_files:
            with open(os.path.join(video_path, annotation_file), 'r') as f:
                data = json.load(f)
            
            for annotation in data["annotations"]:
                image_id = annotation["image_id"]
                if image_id not in all_annotations:
                    all_annotations[image_id] = []
                all_annotations[image_id].append((annotation_file, annotation))

        # draw all annotations on each frame and write to video
        for image_id, image_file in frames_to_write.items():
            if image_id in all_annotations:
                self.cv2_img.set_image()
                image = cv2.imread(image_file)

                for annotation_file, annotation in all_annotations[image_id]:

                    # drawing bbox
                    if annotation_file == "bbox_annotations.json":
                        cv2.rectangle(image, (annotation["bbox"][0], annotation["bbox"][1]), 
                                            (annotation["bbox"][2], annotation["bbox"][3]), 
                                            self.annotation_colors[annotation["object_id"]], 2)
                                            
                    # drawing pose
                    elif annotation_file == "pose_annotations.json":
                        for keypoint_annotation in annotation["keypoints"]:
                            if len(keypoint_annotation) != 0:
                                cv2.circle(image, (keypoint_annotation[1][0], keypoint_annotation[1][1]), 
                                                5, self.annotation_colors[annotation["object_id"]], -1)
                                

                        # drawing lines to the keypoints
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
                                cv2.line(image, pt1, pt2, self.annotation_colors[annotation["object_id"]], thickness=1)
        
                video.write(image)

        video.release()

        print("Video has been created called output_video")

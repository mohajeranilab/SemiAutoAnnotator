# SemiAutoAnnotator
## Description

This annotation labeler uses the concept of active learning.
The user will start annotating images but will be assisted by a model that provides annotations. After significant annotations are made, the user can then be prompted to retrain the model that was helping the user annotate to improve its performance. This system drastically reduces the amount of time a user spends annotating. 

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/mohajeranilab/SemiAutoAnnotator.git
    
2. Install depedences:
    ```bash
    pip install -r requirements.txt

## Usage
Run the annotation_labeler.py file, it will give you two file explorer windows to select the video file you want to annotate (it will extract frames for you), and the model/weights file to help you annotate

"In the 'used_videos/' directory, a folder will be created for each selected video, containing the video file itself along with subfolders for extracted frames and JSON annotations. The structure will look like this:
```plaintext
used_videos/
└── video_name_folder/
    ├── extracted_frames/
        └── img_0.jpg
    ├── video_name.mp4
    ├── bbox_annotations.json
    └── pose_annotations.json
```
The annotations created in the json files are very similar to the COCO dataset format. 

When the program runs and an image is open, if the model is able to detect bounding boxes, you will have to assign the ID to the box. Simply click in the middle of the box to select the ID, pressing "N" to increase the ID if needed.


The following keypresses provide different results when the image window is open:
- “B”: Bounding Box Mode. Press “B” to begin drawing a bounding box around an object (click, drag, and release).
  - “F”. Annotation Feces. After entering Bounding Box Mode, you can press “F” to annotate for feces.
  - “H”. Hidden Annotations. After entering Bounding Box Mode, you can press “H” to start a hidden annotation for certain objects that are hidden or out of the screen. 
- “P”. Pose Mode. Once it is in pose mode, you can press on numbers 1-3 for different parts of the mouse to annotate and place down keypoints.  
- “D”. Deletes Annotations. Press D to delete all annotations for the current image.
- “Ctrl+Z”. Undo. Press Ctrl+Z to undo the last annotation, if there are no annotations in the current image, go back to the previous image.
- “Enter”. Next Image. Press Enter to go to the next image of the dataset.
- “Delete”. Last Image. Press Delete to go back to the previous image of the dataset.
- “Esc”. Exit. Press Esc to exit out of the program. 
- “N”. Next Object ID. Press “N” to get the next object ID, the ID should be displayed at the top of the screen. 
- “R”. Retrain. When “R” is pressed, a file explorer window will appear and the user can select the video folders they want to train on if it has annotated frames. the model will be retrained using the current annotations made. It will exit the program and enter the training mode. 
- “V”. Make Video. When “V” is pressed, a file explorer window will appear and the user can select the video folder they want to create a clip with the annotations made for that video, the clip will be saved as “output_video.mp4”.
- “M”. Toggle Model. When “M” is pressed, the model will either be turned off or on, the default is set to on.




Once retraining starts, another folder will be created called labeler_dataset/, it contains further subfolders with images and labels in YOLO training format. The structure will look like this:
```plaintext
labeler_dataset/
└── images/
    └── train/
        └── img_0.jpg
        └── img_1.jpg
    ├── val/
        └── img_2.jpg
├── labels/
    └── train/
        └── img_0.txt
        └── img_1.txt
    ├── val/
        └── img_2.txt
```


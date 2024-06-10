# SemiAutoAnnotator
## Description

This annotation labeler uses the concept of active learning.
The user will start annotating images but will be assisted by a model that provides annotations. After significant annotations are made, the user can then be prompted to retrain the model that was helping the user annotate to improve its accuracy. This system drastically reduces the amount of time a user spends annotating. 

## Installation
1. Clone the repository:
    git clone https://github.com/mohajeranilab/SemiAutoAnnotator.git
    
2. Install depedences:
    pip install -r requirements.txt

## Usage
Run the annotation_labeler.py file, it will give you two file explorer windows to select the video file you want to annotate (it will extract frames for you), and the model/weights file to help you annotate

"In the 'used_videos/' directory, a folder will be created for each selected video, containing the video file itself along with subfolders for extracted frames and JSON annotations. The structure will look like this:
used_videos/
└── video_name_folder/
    ├── extracted_frames/
        └── img_0.jpg
    ├── video_name.mp4
    ├── bbox_annotations.json
    └── pose_annotations.json

The annotations created in the json files are very similar to the COCO dataset format. 

Once retraining starts, another folder will be created called labeler_dataset/, it contains further subfolders with images and labels in YOLO training format. The structure will look lke this:
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
   
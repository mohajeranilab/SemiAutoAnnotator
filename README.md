# SemiAutoAnnotator
## Description

This annotation labeler uses the concept of active learning and transfer learning to improve the efficiency of the annotation process.
The user will start annotating images and will be assisted by a model that provides suggested annotations. After significant annotations are made, the user can then be prompted to retrain the assisting model to improve its performance. Each retraining session uses the current model as a base, incorporating its pre-trained weights as a starting point for the retraining. This iterative approach significantly reduces the time required for manual annotation by continuously refining the model's accuracy.


## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/mohajeranilab/SemiAutoAnnotator.git
    
2. Install dependencies:
    ```bash
    pip install -r requirements_gpu.txt
    ```
    or if you do not have a GPU and would like to use the annotation tool without the assistance of a model
    ```bash
    pip install -r requirements.txt

## Usage
Run the annotation_labeler.py file, three optional arguments can be passed,
```bash
--frame_skip= (default: 50) --model_path= (default: None) --clustering=(default: False)
``` 
It is NECESSARY to specify a path to the model/weights file if you want the assistance of a model AND make use of the clustering. Aftewards, the program will give you a file explorer window to select the video file you want to annotate (it will extract frames for you).


"In the 'used_videos/' directory, a folder will be created for each selected video, containing the video file itself along with subfolders for extracted frames and JSON annotations. The structure will look like this:
```plaintext
used_videos/
└── video_name_folder/
    ├── extracted_frames/
        └── img_0.jpg
        └── ...
    ├── video_name.mp4
    ├── bbox_annotations.json
    └── pose_annotations.json
```
The annotations created in the json files are very similar to the COCO dataset format. 

<!-- When the program runs and an image is open, if the model is able to detect bounding boxes, you will have to assign the ID to the box. Simply click in the middle of the box to select the ID, pressing "N" to increase the ID if needed. -->
You can either press the buttons on the PyQt GUI on the side, or by pressing the specific keybinds. The following key and button presses provide different results when the image window is SELECTED.

- “B”: Bounding Box Mode. Begin drawing a bounding box around an object (click, drag, and release).
  - “F”. Annotation Feces. After entering Bounding Box Mode, you can press “F” to annotate for feces.
  - “H”. Hidden Annotations. After entering Bounding Box Mode, you can press “H” to start a hidden annotation for certain objects that are hidden or out of the screen. 
- “P”. Pose Mode. Once it is in pose mode, you can press on numbers 1-7 for different parts of the mouse to annotate and place down keypoints.  
  - "1". Head 
  - "2". Tail
  - "3". Neck
  - "4". R Hand
  - "5". L Hand
  - "6". R Leg
  - "7". L Leg
- “D”. Deletes Annotations. Deletes all annotations for the current image.
- “Ctrl+Z”. Undo. Undo the last annotation, if there are no annotations in the current image, go back to the previous image.
- “Enter”. Next Image. Goes to the next image of the dataset.
- “Backspace/Delete”. Last Image. Goes to the previous image of the dataset.
- “Esc”. Exit. Exits out of the program. 
- “N”. Increment Object ID. Increase the object ID by 1.
- "J". Decrement Object ID. Decrease the object ID by 1. 
- “R”. Retrain. A file explorer window will appear and the user can select the video folders they want to train on ONLY IF it has annotated frames. The model being used currently will be used as pretrained weights during training. After the training process by YOLO is completed, it will exit out of the program.
- “V”. Make Video. A file explorer window will appear and the user can select the video folder they want to create a clip with the annotations made for that video, the clip will be saved as “output_video.mp4”.
- “M”. Toggle Model. The model will either be turned off or on, the default is set to on. If no model is given when starting the program, it will be set to off permanently.
- "E". Editing Mode. The user will now be able to edit the annotations made. To edit the bbox annotations, click and drag one of the four squares



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

## Important Notes
- This code can not be ran using WSL1/2 due to the lack of native GUI support. It may work using an X-Server but this has not been tested yet.
<!-- - Python3.9 or above is needed to run this program (to utilize PyQt). -->
- The PyQt GUI window of buttons can be closed.
# Face_Emotion_Detection




<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Installation](#installation)
* [Usage](#usage)
* [Roadmap](#roadmap)



<!-- ABOUT THE PROJECT -->
## About The Project



This projects detect human face emotion with input of image, video or real-time detection. We are using opencv to detect the face and crop the face into ResNet101 built with PyTorch. 
`github_username`, `repo_name`, `twitter_handle`, `email`


### Built With

* Python3.6
* OpenCV
* Pytorch



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.


### Installation

1. Clone the repo
```git
git clone https://github.com/KTAN119/Face_Emotion_Detection
```
2. Install NPM packages
```pip
pip install -r requirement.txt
```



<!-- USAGE EXAMPLES -->
## Usage

1. To Detect Image
```
python emotion_detection.py --file_type 'image' --img_file INPUT_IMAGE_FILE_PATH --output_image_file OUTPUT_IMAGE_DIRECTORY_PATH --weight RESNET101_WEIGHT_PATH --cascade_file CASCADE_FILE_PATH
```
2. To Detect Video
```
python emotion_detection.py --file_type 'video' --video_file INPUT_VIDEO_FILE_PATH --output_video_file OUTPUT_VIDEO_DIRECTORY_PATH --weight RESNET101_WEIGHT_PATH --cascade_file CASCADE_FILE_PATH
```
3. Real-time Detection
```
python emotion_detection.py --file_type 'real-time' --weight RESNET101_WEIGHT_PATH --cascade_file CASCADE_FILE_PATH
```





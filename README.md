<!-- omit in toc -->
# Face Emotion Detection


<!-- TABLE OF CONTENTS -->
<!-- omit in toc -->
## Table of Contents

- [About The Project](#about-the-project)
  - [Built With](#built-with)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Required files](#required-files)
- [Usage](#usage)
- [Credits](#credits)


<!-- ABOUT THE PROJECT -->
## About The Project

This projects detect human face emotion with input of image, video or real-time detection. We are using OpenCV to detect and crop the face, and using ResNet101/VGG19 built & trained with PyTorch to predict the face"s emotion


### Built With

* Python3.6
* OpenCV
* PyTorch


<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.


### Installation

1. Clone the repo
```bash
git clone https://github.com/KTAN119/Face_Emotion_Detection
```
2. Install required packages
```bash
pip install -r requirement.txt
```

### Required files

The following files are required to run the program:
* Cascade file. The cascade file can be obtained from [here](/haarcascade_frontalface_alt2.xml) or [here](https://github.com/opencv/opencv/tree/master/data/haarcascades)
* PyTorch weight file. The weights for a trained emotion classifier is required to predict the faces" emotions. This file is not provided due to the large file size.


<!-- USAGE EXAMPLES -->
## Usage

1. To Detect Image
```
python emotion_detection.py --input_type "image" --img_file INPUT_IMAGE_FILE_PATH --output_image_directory OUTPUT_IMAGE_DIRECTORY_PATH --weight RESNET101_WEIGHT_PATH --cascade_file CASCADE_FILE_PATH
```
2. To Detect Video
```
python emotion_detection.py --input_type "video" --video_file INPUT_VIDEO_FILE_PATH --output_video_directory OUTPUT_VIDEO_DIRECTORY_PATH --weight RESNET101_WEIGHT_PATH --cascade_file CASCADE_FILE_PATH
```
3. Real-time Detection
```
python emotion_detection.py --input_type "real-time" --weight RESNET101_WEIGHT_PATH --cascade_file CASCADE_FILE_PATH
```

## Credits

Done by [Tan Kim Wai](https://github.com/ktan119) and [Melvin Kok](https://github.com/melvinkokxw)

Sample image from Jesse Burke
Sample video from [here](https://github.com/intel-iot-devkit/sample-videos)
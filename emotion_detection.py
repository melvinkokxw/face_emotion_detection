import argparse
import os

import cv2
import ffmpeg
import torch
import torch.nn as nn
from PIL import Image
from pytorchcv.model_provider import get_model
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument("--file_type", type=str, default="real-time",
                    help="image / video / real-time")
parser.add_argument("--video_file", type=str, default="face_video.mp4",
                    help="input video file path")
parser.add_argument("--img_file", type=str, default="face_image.jpg",
                    help="input image file path")
parser.add_argument("--output_video_directory", type=str, default="output_video",
                    help="output video file path")
parser.add_argument("--output_image_directory", type=str, default="output_image",
                    help="output image file path")
parser.add_argument("--weight", type=str, default="weights/vgg19.pth",
                    help="resnet101 weight path")
parser.add_argument("--cascade_file", type=str, default="haarcascade_frontalface_alt2.xml",
                    help="haar cascade file path")
parser.add_argument("--classifier", type=str, default="vgg19",
                    help="classifier type. resnet101 / vgg19")
opt = parser.parse_args()
print(opt)


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Load pretrained network as backbone
        pretrained = get_model(opt.classifier, pretrained=True)
        self.backbone = pretrained.features
        self.output = pretrained.output
        self.classifier = nn.Linear(1000, 7)

        del pretrained

    def forward(self, x):
        x = self.backbone(x)
        x = x.reshape(x.size(0), -1)
        x = self.output(x)
        x = self.classifier(x)

        return x


test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def preprocess(image):
    image = Image.fromarray(image).convert("RGB")  # Webcam frames are numpy array format
    # Therefore transform back to PIL image
    image = test_transform(image)
    image = image.float()
    #image = Variable(image, requires_autograd=True)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.unsqueeze(0)
    return image

def check_rotation(video_file):
    """Checks if given video file has rotation metadata

    Used as workaround for OpenCV bug 
    https://github.com/opencv/opencv/issues/15499, where the rotation metadata 
    in a video is not used by cv2.VideoCapture. May have to be removed/tweaked 
    when bug is fixed in a new opencv-python release.

    Parameters
    ----------
    video_file : str
        Path to video file to be checked

    Returns
    -------
    rotate_code : enum or None
        Flag fror cv2.rotate to decide how much to rotate the image. 
        None if no rotation is required
    """    
    meta_dict = ffmpeg.probe(video_file)
    rotation_angle = int(meta_dict["streams"][0]["tags"]["rotate"])

    rotate_code = None
    if rotation_angle == 90:
        rotate_code = cv2.ROTATE_90_CLOCKWISE
    elif rotation_angle == 180:
        rotate_code = cv2.ROTATE_180
    elif rotation_angle == 270:
        rotate_code = cv2.ROTATE_90_COUNTERCLOCKWISE

    return rotate_code

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel().to(device)
if torch.cuda.is_available():
    state_dict = torch.load(opt.weight)
else:
    state_dict = torch.load(opt.weight, map_location=device)
model.load_state_dict(state_dict)

emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

face_cascade = cv2.CascadeClassifier(opt.cascade_file)

if opt.file_type == "video":
    os.makedirs(opt.output_video_directory, exist_ok=True)
    cap = cv2.VideoCapture(opt.video_file)
    rotate_code = check_rotation(opt.video_file)

    counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if rotate_code is not None: # Rotate frame if necessary
                frame = cv2.rotate(frame, rotate_code)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.5, minNeighbors=5)
            for(x, y, w, h) in faces:
                #            print(x, y, w, h)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]

                img = preprocess(roi_gray).to(device)
                with torch.no_grad():
                    logits = model(img)
                    prediction = torch.argmax(logits, dim=1)

                font = cv2.FONT_HERSHEY_SIMPLEX
                emotion = emotions[prediction]
                color = (255, 0, 0)
                stroke = 2
                cv2.putText(frame, emotion, (x, y), font,
                            1, color, stroke, cv2.LINE_AA)

                end_cord_x = x+w
                end_cord_y = y+h
                cv2.rectangle(frame, (x, y), (end_cord_x,
                                            end_cord_y), color, stroke)

            img_item = os.path.join(opt.output_video_directory, f"{counter}.png")
            cv2.imwrite(img_item, frame)

            counter += 1
        
        else:
            cap.release()
            cv2.destroyAllWindows()
            break

    img_array = []
    size = set()
    for i in range(counter):
        filename = os.path.join(opt.output_video_directory, f"{i}.png")
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(os.path.join(
        opt.output_video_directory, "output_vid.mp4"), cv2.VideoWriter_fourcc(*"MP4V"), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

elif opt.file_type == "image":
    os.makedirs(opt.output_image_directory, exist_ok=True)
    frame = cv2.imread(opt.img_file)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.5, minNeighbors=5)
    for(x, y, w, h) in faces:
        #            print(x, y, w, h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        img = preprocess(roi_gray).to(device)
        with torch.no_grad():
            logits = model(img)
            prediction = torch.argmax(logits, dim=1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        emotion = emotions[prediction]
        color = (255, 0, 0)
        stroke = 2
        cv2.putText(frame, emotion, (x, y), font,
                    1, color, stroke, cv2.LINE_AA)

        color = (0, 0, 255)
        stroke = 2
        end_cord_x = x+w
        end_cord_y = y+h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

    img_item = os.path.join(opt.output_image_directory, "output_img.jpg")
    cv2.imwrite(img_item, frame)

elif opt.file_type == "real-time":
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.5, minNeighbors=5)
        for(x, y, w, h) in faces:
            # print(x, y, w, h)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            img = preprocess(roi_gray).to(device)
            with torch.no_grad():
                logits = model(img)
                prediction = torch.argmax(logits, dim=1)

            emotion = emotions[prediction]
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (255, 0, 0)
            stroke = 2
            cv2.putText(frame, emotion, (x, y), font,
                        1, color, stroke, cv2.LINE_AA)

            end_cord_x = x+w
            end_cord_y = y+h
            cv2.rectangle(frame, (x, y), (end_cord_x,
                                        end_cord_y), color, stroke)

        cv2.imshow("frame", frame)
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

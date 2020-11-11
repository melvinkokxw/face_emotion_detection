import argparse
from PIL import Image
from pytorchcv.model_provider import get_model
from torchvision import transforms
import torchvision
from torch.utils.data import random_split, DataLoader, Dataset
import torch.nn as nn
import torch
import numpy as np
import os
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--file_type", type=str, default=None,
                    help="image/video/real-time")
parser.add_argument("--video_file", type=str, default='/home/students/acct2014_04/DIP/input_video',
                    help="input video file path")
parser.add_argument("--img_file", type=str, default='/home/students/acct2014_04/DIP/input_image',
                    help="input image file path")
parser.add_argument("--output_video_file", type=str, default='/home/students/acct2014_04/DIP/output_video',
                    help="output video file path")
parser.add_argument("--output_image_file", type=str, default='/home/students/acct2014_04/DIP/output_image',
                    help="output image file path")
parser.add_argument("--weight", type=str, default='/home/students/acct2014_04/DIP/CNN_Weight_Original_Clear_Resnet101/weights_epoch_29_acc_0.6388966285873502.pth',
                    help="resnet101 weight path")
parser.add_argument("--cascade_file", type=str, default='/home/students/acct2014_04/DIP/haarcascade_frontalface_alt2.xml',
                    help="haar cascade file path")
opt = parser.parse_args()
print(opt)


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Load pretrained network as backbone
        pretrained = get_model('resnet101', pretrained=True)
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
    image = Image.fromarray(image).convert(
        'RGB')  # Webcam frames are numpy array format
    # Therefore transform back to PIL image
    image = test_transform(image)
    image = image.float()
    #image = Variable(image, requires_autograd=True)
    image = image.cuda()
    # I don"t know for sure but Resnet-50 model seems to only
    image = image.unsqueeze(0)
    # accpets 4-D Vector Tensor so we need to squeeze another
    return image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNModel().to(device)
state_dict = torch.load(opt.weight)
model.load_state_dict(state_dict)

emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

face_cascade = cv2.CascadeClassifier(opt.cascade_file)

if opt.file_type == 'video':
    os.makedirs(opt.output_video_file, exist_ok=True)
    counter = 0
    cap = cv2.VideoCapture(opt.video_file)
    _, frame = cap.read()
    if frame is not None:
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

        img_item = os.path.join(opt.output_video_file, f"{counter}.png")
        cv2.imwrite(img_item, frame)
        # cv2.imshow("frame", frame)
        # if cv2.waitKey(20) & 0xFF == ord("q"):
        #     break

        counter += 1
    cap.release()
    cv2.destroyAllWindows()

    img_array = []
    no_img = len(os.listdir(opt.output_video_file))
    for counter in range(no_img):
        filename = os.path.join(opt.output_video_file, f"{counter}.png")
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(os.path.join(
        opt.output_video_file, "output_vid.mp4"), cv2.VideoWriter_fourcc(*'MP4V'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

elif opt.file_type == 'image':
    os.makedirs(opt.output_image_file, exist_ok=True)
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

    img_item = os.path.join(opt.output_image_file, "output_img.jpg")
    cv2.imwrite(img_item, frame)

elif file_type == 'real-time':
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.5, minNeighbors=5)
        for(x, y, w, h) in faces:
            print(x, y, w, h)
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

        cv2.imshow('frame', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

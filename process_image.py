"""
Detect dogs in real time and discriminate breeds.

Usage: python3 app.py [label=path/to/label] [model=path/to/model] [video=path/to/video]
"""

import argparse
import shutil
from datetime import datetime
import time
import os
import sys

import cv2
from keras.models import load_model
import numpy as np
from PIL import Image
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets

# 데이터 분포 시각화 패키지
import seaborn as sns

# db의 한 종류
from dynamodb.dynamodb import Dynamodb
from model.model import DetectDogs

INPUT_SIZE = 299

model = None
model_path = None
label_path = None
icon_path = './design/icon.png'
splash_pix_path = './design/splash_pix.jpg'


class VideoCaptureView(DetectDogs):
    repeat_interval = 33  # ms

    def __init__(self, input_path=None):
        super(VideoCaptureView, self).__init__()

        self.pixmap = None
        self.item = None
        self.rect_items = []

        self.breeds = []
        global label_path
        with open(label_path) as f:
            lines = f.readlines()
            for line in lines:
                self.breeds.append(line.replace('\n', ''))

        color_palette = np.array(sns.color_palette(n_colors=len(self.breeds)))
        self.colors = color_palette * 255

        if input_path:
            self.capture = cv2.imread(input_path)
        else:
            print("no input path")
            exit()

        self.set_video_image(input_path)



    def process_image(self, frame, coordinates):
        """
        Process image.

        Draw rectangle and text.

        Args:
            frame (np.ndarray): BGR image
            coordinates (list): Coordinates of dogs

        Returns:
            frame (np.ndarray): BGR image after processing
        """

        for coordinate in coordinates:
            frame_crop = frame[coordinate[1]:coordinate[3],
                         coordinate[0]:coordinate[2]]

            frame_crop = cv2.resize(frame_crop, (INPUT_SIZE, INPUT_SIZE)) / 255

            frame_crop = np.reshape(frame_crop, (1, INPUT_SIZE, INPUT_SIZE, 3))
            prediction = model.predict(frame_crop)
            idx = np.argmax(prediction)
            self.breed_label = self.breeds[idx]

            color = tuple(map(int, self.colors[idx]))

            # Draw label.
            x = coordinate[0] + 10
            y = coordinate[1] - 10 if coordinate[1] - \
                                      20 > 20 else coordinate[1] + 20
            cv2.putText(
                img=frame,
                text="",
                org=(
                    x,
                    y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA)

            # Draw renctangle.
            frame = cv2.rectangle(frame,
                                  (coordinate[0], coordinate[1]),
                                  (coordinate[2], coordinate[3]),
                                  color,
                                  thickness=4)

        return frame, self.breed_label

    def set_video_image(self, input_path):

        count = 0

        while True:

            image_paths = []

            # input_path 에 file 이름을 넣지 않고 디렉토리까지만 지정했을 경우
            if os.path.isdir(input_path):
                # os.listdir : 이 디렉토리에 있는 전체 파일의 이름을 리스트 형태로 반환한다
                for inp_file in os.listdir(input_path):
                    image_paths += [input_path + inp_file]

            # 그중에서 jpg, png, jpeg 확장자를 가진 파일만 남긴다
            image_paths = [inp_file for inp_file in image_paths
                           if (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]

            # the main loop
            for image_path in image_paths:
                frame = cv2.imread(image_path)

                breed_label = ''

                try:
                    shape = frame.shape
                    height, width, dim = shape

                    bytes_per_line = dim * width

                    # Detect dogs
                    frame_pil = Image.fromarray(
                        np.uint8(
                            cv2.cvtColor(
                                frame,
                                cv2.COLOR_BGR2RGB)))

                    coordinates = self.detect_image(frame_pil)

                    filename = image_path.split('/')[-1]

                    # 강아지를 발견했다면 -> 강아지 위에 박스처리를 한 후 가공된 사진을 저장한다
                    if len(coordinates)>0:
                        count += 1
                        print("count = "+str(count))
                        frame, breed_label = self.process_image(frame, coordinates)
                        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # output 폴더에 가공된 사진을 저장한다
                        # write the image with bounding boxes to file
                        cv2.imwrite("./predict_output/"+breed_label+"__"+filename, np.uint8(frame))
                    else:
                        print("It's not a dog")

                    # 원본 사진을 originals 폴더로 옮긴다(강아지 발견 여부와 무관)
                    shutil.move(image_path, "./originals/"+filename)

                except AttributeError:
                    print("Attribute Error")


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    parser.add_argument(
        '--label',
        default='model/breeds_16.txt',
        type=str,
        help='path to classification label (default: model_16.txt)')

    parser.add_argument(
        '--model',
        default='model/classification/model_16.h5',
        type=str,
        help='path to classification model (default: model_16.h5)')

    parser.add_argument(
        '--video',
        default=False,
        type=str,
        help='path to video to use instead of web camera')

    FLAGS = parser.parse_args()

    global label_path
    label_path = FLAGS.label

    global model_path
    model_path = FLAGS.model


    input_path = FLAGS.video

    global model
    model = load_model(model_path)

    video_capture_view = VideoCaptureView(input_path=input_path)


if __name__ == "__main__":
    main()

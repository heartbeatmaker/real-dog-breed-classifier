"""
Detect dogs in real time and discriminate breeds.

실행법: python process_image.py --video ./screenshot_output/
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

# 데이터 분포 시각화 패키지
import seaborn as sns

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

        # 분류 기준으로 삼고자 하는 견종을 배열에 담는다
        self.breeds = []
        global label_path
        with open(label_path) as f:
            lines = f.readlines()
            for line in lines:
                self.breeds.append(line.replace('\n', ''))

        color_palette = np.array(sns.color_palette(n_colors=len(self.breeds)))
        self.colors = color_palette * 255

        # 디렉토리에 있는 사진에서 강아지를 찾는다 + 찾은 강아지의 견종을 찾는다 + 강아지 주변에 박스처리를 한다
        self.detect_dog_and_find_its_breed(input_path)

    # 찾은 강아지 주변에 박스처리를 한다
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
            # 강아지가 있는 부분만 자른다
            frame_crop = frame[coordinate[1]:coordinate[3],
                         coordinate[0]:coordinate[2]]

            frame_crop = cv2.resize(frame_crop, (INPUT_SIZE, INPUT_SIZE)) / 255

            frame_crop = np.reshape(frame_crop, (1, INPUT_SIZE, INPUT_SIZE, 3))

            # 찾은 강아지의 견종을 찾는다
            prediction = model.predict(frame_crop)
            idx = np.argmax(prediction)
            self.breed_label = self.breeds[idx]

            # 견종에 따라 박스 선 색깔을 다르게 한다
            color = tuple(map(int, self.colors[idx]))

            # Draw label.
            x = coordinate[0] + 10
            y = coordinate[1] - 10 if coordinate[1] - \
                                      20 > 20 else coordinate[1] + 20
            cv2.putText(
                img=frame,
                text=self.breed_label,
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

        # 박스처리된 사진과, 견종을 반환한다
        return frame, self.breed_label

    # 디렉토리에 있는 사진에서 강아지를 찾는다 + 찾은 강아지의 견종을 찾는다 + 강아지 주변에 박스처리를 한다
    def detect_dog_and_find_its_breed(self, input_path):

        count = 0

        while True:

            image_paths = []

            # input 디렉토리에서 사진파일을 가져온다
            if os.path.isdir(input_path):
                # os.listdir : 이 디렉토리에 있는 전체 파일의 이름을 리스트 형태로 반환한다
                for inp_file in os.listdir(input_path):
                    image_paths += [input_path + inp_file]

            # 그중에서 jpg, png, jpeg 확장자를 가진 파일만 남긴다
            image_paths = [inp_file for inp_file in image_paths
                           if (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]

            # the main loop - 사진을 한 장씩 분석한다
            for image_path in image_paths:

                filename = image_path.split('/')[-1]

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

                    # 사진에서 강아지를 찾고, 찾은 강아지의 좌표를 구한다
                    coordinates = self.detect_image(frame_pil)

                    # 강아지를 발견했다면 -> 강아지 위에 박스처리를 한 후 가공된 사진을 저장한다
                    if len(coordinates)>0:
                        count += 1
                        print("count = "+str(count))

                        # 강아지 위에 박스처리를 한다
                        frame, breed_label = self.process_image(frame, coordinates)
                        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # output 폴더에 가공된 사진을 저장한다
                        # 파일 이름 : 견종__원본파일이름.jpg
                        # write the image with bounding boxes to file
                        cv2.imwrite("./predict_output/"+breed_label+"__"+filename, np.uint8(frame))
                    else:
                        # 사진에서 강아지를 발견하지 않았을 경우
                        print("It's not a dog")

                    # 원본 사진을 originals 폴더로 옮긴다(강아지 발견 여부와 무관)
                    shutil.move(image_path, "./originals/"+filename)

                except AttributeError:
                    # 코드 작동 중, 빈 디렉토리에 사진이 갑자기 추가되면, shape 함수에서 에러가 발생하는 경우가 있다
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

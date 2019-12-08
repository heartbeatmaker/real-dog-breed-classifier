import numpy as np
import os
import cv2
from .colors import get_color

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        
        self.c       = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
            
        return self.score      

def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3    

def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])  
    
    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    
    union = w1*h1 + w2*h2 - intersect
    
    return float(intersect) / union

def draw_boxes(image, boxes, labels, obj_thresh, quiet=True):
    for box in boxes:
        label_str = ''
        label = -1
        
        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:  # 60% 이상의 확률인지 확인
                if label_str != '': label_str += ', '
                label_str += (labels[i] + ' ' + str(round(box.get_score()*100, 2)) + '%')  # ex) dog 99.98%
                label = i
            if not quiet: print(label_str)
                
        if label >= 0:
            text_size = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 1.1e-3 * image.shape[0], 5)
            width, height = text_size[0][0], text_size[0][1]
            region = np.array([[box.xmin-3,        box.ymin], 
                               [box.xmin-3,        box.ymin-height-26], 
                               [box.xmin+width+13, box.ymin-height-26], 
                               [box.xmin+width+13, box.ymin]], dtype='int32')  

            cv2.rectangle(img=image, pt1=(box.xmin,box.ymin), pt2=(box.xmax,box.ymax), color=get_color(label), thickness=5)
            cv2.fillPoly(img=image, pts=[region], color=get_color(label))
            cv2.putText(img=image, 
                        text=label_str, 
                        org=(box.xmin+13, box.ymin - 13), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=1e-3 * image.shape[0], 
                        color=(0,0,0), 
                        thickness=2)
        
    return image


# def process_image(self, frame, coordinates):
#
#     INPUT_SIZE = 299
#
#     for coordinate in coordinates:
#         frame_crop = frame[coordinate[1]:coordinate[3],
#                      coordinate[0]:coordinate[2]]
#
#         frame_crop = cv2.resize(frame_crop, (INPUT_SIZE, INPUT_SIZE)) / 255
#
#         frame_crop = np.reshape(frame_crop, (1, INPUT_SIZE, INPUT_SIZE, 3))
#         prediction = model.predict(frame_crop)
#         idx = np.argmax(prediction)
#         self.breed_label = self.breeds[idx]


def draw_boxes_for_dogs(model, image, boxes, labels, breed_labels, obj_thresh, quiet=True):

    INPUT_SIZE = 299

    for box in boxes:
        label_str = ''
        label = -1

        isDog = False

        for i in range(len(labels)):
            if box.classes[i] > obj_thresh and "dog" == labels[i]:  # 60% 이상의 확률로 개인지 확인

                score = round(box.get_score()*100, 2)
                # label_str += (labels[i] + ' ' + str(score) + '%')  # 박스 위에 표시할 정보: 라벨, 확률
                label_str += (labels[i])  # 박스 위에 표시할 정보: only 라벨
                label = i
                isDog = True

            if not quiet: print(label_str)

        if label >= 0 and isDog:  # 개 주변에만 박스를 그린다

            frame_crop = image[box.xmin:box.ymin,
                         box.xmax:box.ymax]

            frame_crop = cv2.resize(frame_crop, (INPUT_SIZE, INPUT_SIZE)) / 255

            frame_crop = np.reshape(frame_crop, (1, INPUT_SIZE, INPUT_SIZE, 3))
            prediction = model.predict(frame_crop)
            idx = np.argmax(prediction)
            breed_label = breed_labels[idx]
            label_str = breed_label


            text_size = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 1.1e-3 * image.shape[0], 5)
            width, height = text_size[0][0], text_size[0][1]
            region = np.array([[box.xmin-3,        box.ymin],
                               [box.xmin-3,        box.ymin-height-26],
                               [box.xmin+width+13, box.ymin-height-26],
                               [box.xmin+width+13, box.ymin]], dtype='int32')

            cv2.rectangle(img=image, pt1=(box.xmin,box.ymin), pt2=(box.xmax,box.ymax), color=get_color(label), thickness=5)
            cv2.fillPoly(img=image, pts=[region], color=get_color(label))
            cv2.putText(img=image,
                        text=label_str,
                        org=(box.xmin+13, box.ymin - 13),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1e-3 * image.shape[0],
                        color=(0,0,0),
                        thickness=2)

    return image

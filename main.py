import shutil
import os
import time
import argparse
import cv2
import subprocess as sp
import ultralytics

# get models from https://github.com/ultralytics/ultralytics

MODEL_SIZE = 'm'
DETECTION_MODEL = f"yolov8{MODEL_SIZE}.pt"
CLASSIFICATION_MODEL = f"yolov8{MODEL_SIZE}-cls.pt"



def main():
    # https://docs.ultralytics.com/modes/predict/#probs
    # https://www.geeksforgeeks.org/python-opencv-cv2-imwrite-method/
    model = ultralytics.YOLO(DETECTION_MODEL, task='detect')
    results = model('img.png')
    for r in results:
        plot = r.plot()
        cv2.imwrite('plot.png', plot)

if __name__ == '__main__':
    main()

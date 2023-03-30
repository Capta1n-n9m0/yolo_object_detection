import shutil
import os
import time
import argparse

from pprint import pprint
import numpy as np
import ultralytics
from PIL import Image, ImageDraw
from pathlib import Path

# get models from https://github.com/ultralytics/ultralytics

MODEL_SIZE = 'm'
DETECTION_MODEL = f"yolov8{MODEL_SIZE}.pt"
CLASSIFICATION_MODEL = f"yolov8{MODEL_SIZE}-cls.pt"


def main():
	# https://docs.ultralytics.com/modes/predict/#probs
	# https://www.geeksforgeeks.org/python-opencv-cv2-imwrite-method/
	detection_model = ultralytics.YOLO(DETECTION_MODEL, task='detect')
	classification_model = ultralytics.YOLO(CLASSIFICATION_MODEL, task='classify')
	image_name = 'img_1.png'
	# image_path = Path(image_name).absolute()
	# print(image_path.)
	# exit(1)
	
	img = Image.open(image_name)
	results = detection_model(img)
	detections = []
	for r in results:
		for b, c in zip(r.boxes, r.boxes.cls):
			b = b.cpu()
			prob = np.array(b.data)[0, 4]
			bbox = b.xyxy.numpy()[0]
			temp_img = img.crop(bbox)
			detections.append({
				'bbox': bbox.tolist(),
				'category': r.names[c.item()],
				'score': prob,
				'crop': temp_img,
			})
	crops = [d['crop'] for d in detections]
	results = classification_model(crops)
	for r, d in zip(results, detections):
		r = r.cpu()
		temp_id = np.argmax(np.array(r.probs))
		temp_name = r.names[temp_id]
		temp_prob = np.array(r.probs)[temp_id]
		if temp_prob > d['score']:
			d['category'] = temp_name
			d['score'] = temp_prob
	drawer = ImageDraw.Draw(img)
	for d in detections:
		bbox = d['bbox']
		drawer.rectangle(bbox, outline='red')
	
	img.save('result.png')
	
	pprint(detections)



if __name__ == '__main__':
	main()

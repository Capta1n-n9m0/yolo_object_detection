import shutil
import os
import time
import argparse
import subprocess as sp

import numpy as np
import ultralytics
from PIL import Image, ImageDraw

# get models from https://github.com/ultralytics/ultralytics

MODEL_SIZE = 'm'
DETECTION_MODEL = f"yolov8{MODEL_SIZE}.pt"
CLASSIFICATION_MODEL = f"yolov8{MODEL_SIZE}-cls.pt"


def main():
	# https://docs.ultralytics.com/modes/predict/#probs
	# https://www.geeksforgeeks.org/python-opencv-cv2-imwrite-method/
	detection_model = ultralytics.YOLO(DETECTION_MODEL, task='detect')
	classification_model = ultralytics.YOLO(CLASSIFICATION_MODEL, task='classify')
	image_path = 'img.png'
	img = Image.open(image_path)
	results = detection_model(img)
	detections = []
	objects = []
	for r in results:
		objectsId = r.boxes.cls
		for i in objectsId:
			objects.append(r.names[i.item()])
		for i, (b, c) in enumerate(zip(r.boxes, r.boxes.cls)):
			prob = np.array(b.cpu().data)[0, 4]
			b = b.cpu().xyxy.numpy()[0]
			detections.append({
				'bbox': b,
				'category_id': c.item(),
				'category': r.names[c.item()],
				'score': prob
			})
			temp_img = img.crop((b[0], b[1], b[2], b[3]))
			temp_img.save(f"temp_{i}_{objects[i]}.png")
		plot = r.plot(show_conf=True, line_width=1, pil=True)[:, :, ::-1]
		plot = Image.fromarray(plot)
		plot.save('plot.png')
	print(detections)
	crops = [f"temp_{i}_{o}.png" for i, o in enumerate(objects)]
	print(crops)
	results = classification_model(crops)
	objects = []
	for i, r in enumerate(results):
		r = r.cpu()
		objects.append(r.names[np.argmax(r.probs).item()])
	print(objects)



if __name__ == '__main__':
	main()

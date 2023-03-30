# https://docs.ultralytics.com/modes/predict/#probs
# https://www.geeksforgeeks.org/python-opencv-cv2-imwrite-method/
# get models from https://github.com/ultralytics/ultralytics

import sys
from uuid import uuid4
import numpy as np
import ultralytics
from PIL import Image, ImageDraw
from pathlib import Path
import json

MODEL_SIZE = 'm'
HOME_DIR = Path(__file__).parent.absolute()
DETECTION_MODEL = HOME_DIR / f"yolov8{MODEL_SIZE}.pt"
CLASSIFICATION_MODEL = HOME_DIR / f"yolov8{MODEL_SIZE}-cls.pt"
COLORS = [
	(255, 0, 0), # red
	(0, 255, 0), # green
	(0, 0, 255), # blue
	(255, 255, 0), # yellow
	(255, 0, 255), # magenta
	(0, 255, 255), # cyan
	(255, 255, 255), # white
	(0, 0, 0), # black
	(128, 128, 128), # gray
	(128, 0, 0), # maroon
	(128, 128, 0), # olive
	(0, 128, 0), # green
	(128, 0, 128), # purple
	(0, 128, 128), # teal
	(0, 0, 128), # navy
	(255, 165, 0), # orange
	(255, 69, 0), # orangered
	(0,  255, 165), # springgreen
	(0, 255, 69), # greenyellow
	(165, 0, 255), # violet
	(69, 0, 255), # indigo
]

detection_model: ultralytics.YOLO = None
classification_model: ultralytics.YOLO = None

def build_models(detection_model_path=DETECTION_MODEL, classification_model_path=CLASSIFICATION_MODEL):
	global detection_model
	global classification_model
	detection_model = ultralytics.YOLO(detection_model_path, task='detect')
	classification_model = ultralytics.YOLO(classification_model_path, task='classify')

def main(image_input, storage_input):
	if detection_model is None or classification_model is None:
		build_models()
	# Get paths
	image_path = Path(image_input).absolute()
	storage_path = Path(storage_input).absolute()
	
	# Run detection
	img = Image.open(image_path)
	detection_result = detection_model(img)[0]
	detections = []
	for b, c in zip(detection_result.boxes, detection_result.boxes.cls):
		b = b.cpu()
		prob = np.array(b.data)[0, 4]
		bbox = b.xyxy.numpy()[0]
		temp_img = img.crop(bbox)
		detections.append({
			'bbox': bbox.tolist(),
			'category': detection_result.names[c.item()],
			'score': prob,
			'crop': temp_img,
		})
		
	# Run classification on crops
	crops = [d['crop'] for d in detections]
	if len(crops) > 0:
		classification_results = classification_model(crops)
		for r, d in zip(classification_results, detections):
			r = r.cpu()
			temp_id = np.argmax(np.array(r.probs))
			temp_name = r.names[temp_id]
			temp_prob = np.array(r.probs)[temp_id]
			if temp_prob > d['score']:
				d['category'] = temp_name
				d['score'] = temp_prob
		
	# Run classification on the whole image
	classification_result = classification_model(img)[0].cpu()
	temp_id = np.argmax(np.array(classification_result.probs))
	temp_name = classification_result.names[temp_id]
	temp_prob = np.array(classification_result.probs)[temp_id]
	detections.append({
		'bbox'    : [0, 0, img.width, img.height],
		'category': temp_name,
		'score'   : temp_prob,
		'crop'    : img,
	})
	
	# Save results
	colormap = {}
	drawer = ImageDraw.Draw(img)
	for d in detections[:-1]:
		if d['category'] not in colormap:
			colormap[d['category']] = COLORS[len(colormap)]
		drawer.rectangle(d['bbox'], outline=colormap[d['category']], width=1)
	output_name = f"{uuid4()}.png"
	output_path = storage_path / output_name
	img.save(output_path)
	
	# Return results
	output = output_path.name
	result = json.dumps({
		'detections': [{
			'bbox': [float(b) for b in d['bbox']],
			'category': d['category'].replace('_', ' '),
			'score': float(d['score']),
		} for d in detections],
	})
	
	return output, result

if __name__ == '__main__':
	image_input = sys.argv[1]
	storage_input = sys.argv[2]
	build_models()
	output, result = main(image_input, storage_input)
	print(output)
	print(result)

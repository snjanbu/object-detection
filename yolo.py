import argparse
import cv2
import numpy as np
import easygui
import filetype

# Loading yolo framework
def load_yolo():
	# Loading weight and class names
	names_path = "yolo/coco.names"
	cfg_path = "yolo/yolov3.cfg"
	weights_path = "yolo/yolov3.weights"
	class_names = []

	# parse class names
	with open(names_path, 'rt') as f:
		class_names = "".join(f.readlines()).strip().rsplit("\n")

	# # Construct darknet model
	model = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
	layer_names = model.getLayerNames()
	unconnected_layers = model.getUnconnectedOutLayers()
	
	# without cuda support, layer_names is a 1-D array
	output_layers = [layer_names[i - 1] for i in unconnected_layers]
	
	# Random colors for boxes
	colors = np.random.uniform(0, 255, size=(len(class_names), 3))
	return model, class_names, output_layers, colors

# Loading image, or converting img to array
def load_img(path):
	image = cv2.imread(path)
	# Resize to (416, 416) standard yolov3 size
	image = cv2.resize(image, None, fx=0.4, fy=0.4)
	height, width, channels = image.shape
	return height, width, channels, image

# Preprocessing image for yolov3
def preprocess_img(image, model, output_layers):
	# blobfromimage does scaling of 1/255 and mean subtraction of 0 providing an output of size (320, 320)
	blob = cv2.dnn.blobFromImage(image, scalefactor=1/255, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
	model.setInput(blob)
	# outputs will give a nested list of centerx, centery, height, width, probability of class
	outputs = model.forward(output_layers)
	return blob, outputs

# Get bounding box co-ordinates
def get_box_coordinates(outputs, img_height, img_width):
	boxes, conf, class_ids = [], [], []
	for output in outputs:
		for detection in output:
			# Class probabilities will have the probability associated with each class
			class_probabilities = detection[5:]
			max_class_id = np.argmax(class_probabilities)
			max_class_probability = class_probabilities[max_class_id]
			# Selecting confidence greater than 30%
			if max_class_probability > 0.3:
				center_x, center_y = detection[0] * img_width, detection[1] * img_height
				width, height = detection[2], detection[3]
				# Scaling from 0-1 to image size
				width *= img_width
				height *= img_height
				# Subtracting to get top origin point
				top_x = center_x - (width / 2)
				top_y = center_y - (height / 2)
				boxes.append([top_x, top_y, width, height])
				conf.append(max_class_probability)
				class_ids.append(max_class_id)
	return boxes, conf, class_ids


# Draw boxes
def draw_boxes(img, boxes, colors, conf, class_ids, class_names):
	indexes = cv2.dnn.NMSBoxes(boxes, conf, 0.5, 0.3)
	font = cv2.FONT_HERSHEY_SIMPLEX
	for index, box in enumerate(boxes):
		if index in indexes:
			x, y, w, h = list(map(int, box))
			label = str(class_names[class_ids[index]])
			color = colors[class_ids[index]]
			cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
			cv2.putText(img, label, (x, y-5), font, 1, color, 1)
	cv2.imshow("Input", img)

# Show output
def show_output(image, model, output_layers, height, width, colors, class_names):
	blob, outputs = preprocess_img(image, model, output_layers)
	boxes, conf, class_ids = get_box_coordinates(outputs, height, width)
	draw_boxes(image, boxes, colors, conf, class_ids, class_names)
	

model, class_names, output_layers, colors = load_yolo()

file = easygui.fileopenbox()
file_input = filetype.guess(file)
file_type = file_input.mime

if file_type.startswith("image"):
	height, width, channels, image = load_img(file)
	show_output(image, model, output_layers, height, width, colors, class_names)
	while True:
		key = cv2.waitKey(1)
		if key == 27:
			break
elif file_type.startswith("video"):
	cap = cv2.VideoCapture(file)
	while True:
		_, frame = cap.read()
		height, width, channels = frame.shape
		show_output(frame, model, output_layers, height, width, colors, class_names)
		key = cv2.waitKey(1)
		if key == 27:
			break
	cap.release()








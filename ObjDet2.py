# Import packages
import os
import cv2
import numpy as np
import argparse
import sys
import time
import yaml
from threading import Thread
import importlib.util

# Define VideoStream class with separate Thread for smoother video feed
class VideoStream:
    def __init__(self, resolution=(640, 480), framerate=30):
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

# Parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--model', help='Folder where .tflite file is located.', required=True)
parser.add_argument('--graph', help='Name of .tflite file', default='YOLOv8sModel.tflite')
parser.add_argument('--label', help='Name of the labelmap file', default='detect.yaml')
parser.add_argument('--resolution', help='Webcam resolution in WxH.', default='1280x720')
parser.add_argument('--threshold', help='Minimum threshold to display detected objects', default=0.5)

args = parser.parse_args()

MODEL = args.model
GRAPH = args.graph
LABELMAP = args.label
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
min_threshold = float(args.threshold)

wd_path = os.getcwd()
model_path = os.path.join(wd_path, MODEL, GRAPH)
labels_path = os.path.join(wd_path, MODEL, LABELMAP)

# Load the labels from YAML file
with open(labels_path, 'r') as f:
    labels_yaml = yaml.safe_load(f)
    classLabels = labels_yaml['labels']

# Import TensorFlow libraries
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter

# Load TFLite model and allocate tensors
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Output Details:", output_details)

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1
outname = output_details[0]['name']

# Define the output indices based on TensorFlow version used to create the model
if 'StatefulPartitionedCall' in outname:
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

print("Boxes Index:", boxes_idx)
print("Classes Index:", classes_idx)
print("Scores Index:", scores_idx)

# Frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
time.sleep(1)

def detection_results(interpreter, output_details):
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    output_data = output_data.reshape((8400, 12))
    
    boxes = output_data[:, :4]
    classes = output_data[:, 4]
    scores = output_data[:, 5]
    
    return boxes, classes, scores

def draw_detection_boxes(frame, boxes, classes, scores, classLabels, imW, imH, min_conf_threshold=0.5):
    font = cv2.FONT_HERSHEY_SIMPLEX
    txt_thickness = 2
    font_size = 1

    for i in range(len(scores)):
        if (scores[i] > min_conf_threshold) and (scores[i] <= 1.0):
            x_min = int(max(1, (boxes[i][1] * imW)))
            y_min = int(max(1, (boxes[i][0] * imH)))
            x_max = int(min(imW, (boxes[i][3] * imW)))
            y_max = int(min(imH, (boxes[i][2] * imH)))

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (10, 255, 0), 2)

            object_name = classLabels[int(classes[i])]
            label = f'{object_name}: {int(scores[i] * 100)}%'
            labelSize, baseLine = cv2.getTextSize(label, font, 0.7, txt_thickness)
            label_ymin = max(y_min, labelSize[1] + 10)
		
            cv2.rectangle(frame, (x_min, label_ymin - labelSize[1] - 10), 
                          (x_min + labelSize[0], label_ymin + baseLine - 10), 
                          (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (x_min, label_ymin - 7), font, 0.7, (0, 0, 0), txt_thickness)

while True:
    t1 = cv2.getTickCount()
    frame1 = videostream.read()
    
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    boxes, classes, scores = detection_results(interpreter, output_details)

    draw_detection_boxes(frame, boxes, classes, scores, classLabels, imW, imH, min_threshold)
	
    cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('Object Detection', frame)

    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
videostream.stop()

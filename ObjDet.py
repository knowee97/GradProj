# Import packages
import os
import cv2
import numpy as np
import argparse
import sys
import time
from threading import Thread
import importlib.util

#-------------------------------------------------------------------------------------------------------------------------------------------------

# Source Link in Trello
# Define VideoStream class with separate Thread : Smoother video feed
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self, resolution=(640,480), framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()
	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True

#-------------------------------------------------------------------------------------------------------------------------------------------------

# Helps call/find files from the command prompt/terminal
parser = argparse.ArgumentParser()

# Positional Argument
parser.add_argument('--model', help='Folder .tflite file is located.', required=True)				# parser.add_argument('model', help=...)
# Optional Arguments
parser.add_argument('--graph', help='Name of .tflite file', default = 'YOLOv8sModel.tflite')			# new weight
parser.add_argument('--label', help='Name of the labelmap file', default = 'detect.yaml')			# yaml file with 8 classes (Trail)
parser.add_argument('--resolution', help='Webcam resolution in WxH.', default='1280x720')
parser.add_argument('--threshold', help='Minimum threshold to display detected objects', default= 0.5)

# Breaks arguments
args = parser.parse_args()

MODEL= args.model                                    								# MODEL = parser.parser_args().model
GRAPH = args.graph
LABELMAP = args.label
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
min_threshold = float(args.threshold)

#-------------------------------------------------------------------------------------------------------------------------------------------------

CWD_PATH = os.getcwd()						    # Path of current working directory
CKPT_PATH = os.path.join(CWD_PATH, MODEL, GRAPH)                    # Path to file: Current dir to Model dir, finds model file 
LABELS_PATH = os.path.join(CWD_PATH, MODEL, LABELMAP)               # Path to file: Current dir to Model dir, finds label file

#-------------------------------------------------------------------------------------------------------------------------------------------------#

# Load the labels from YAML file
with open(LABELS_PATH, 'r') as f:
    labels_yaml = yaml.safe_load(f)
    classLabels = labels_yaml['labels']
	
#-------------------------------------------------------------------------------------------------------------------------------------------------#

# Import TensorFlow libraries
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter
       
# Load TFLite model
interpreter = Interpreter(model_path = CKPT_PATH)     				 # tflite.Interpreter()

# Needs to be called. TFLite pre-plans tensor allocations to optimize inference
interpreter.allocate_tensors()

# Model details. Each item as a dictionary with details (name, index, shape, dtype) 
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

# Normalization
input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']

#-------------------------------------------------------------------------------------------------------------------------------------------------

indices =
{
    'TF2': (1, 3, 0),
    'TF1': (0, 1, 2)
}

if 'StatefulPartitionedCall' in outname:
    boxes_idx, classes_idx, scores_idx = indices['TF2']
else:
    boxes_idx, classes_idx, scores_idx = indices['TF1']

# Frame Rate Calc Initialized 
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH), framerate=30).start()
time.sleep(1)

while True:
    t1 = cv2.getTickCount() 					# Start timer
    frame1 = videostream.read()					# Grab frame from video stream
    
    frame = frame1.copy()					# Acquire frame and resize to expected shape [1xHxWx3]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)		# Changes to RGB colorway
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if floating model used (model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] 	# Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] 	# Class index of detected objects
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] 	# Confidence of detected objects

#-------------------------------------------------------------------------------------------------------------------------------------------------
	
    # Loop over all detections and draw detection box if confidence is above minimum threshold
    font = cv2.FONT_HERSHEY_SIMPLEX
    txt_thickness = 2
    font_size = 1

    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
	    x_min = int(max(1, (boxes[i][1] * imW)))    
	    y_min = int(max(1, (boxes[i][0] * imH)))
            x_max = int(min(imW, (boxes[i][3] * imW)))
	    y_max = int(min(imH, (boxes[i][2] * imH)))
           
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (10, 255, 0), 2)			# rect around frame from min & max, (R, G, B), and thickness

            # Draw labels
	    # REMINDER: 'lables' changed to 'classLabels'
            object_name = labels[int(classes[i])] 							# Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) 					# Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, font, 0.7, txt_thickness)			# Get font size
            label_ymin = max(y_min, labelSize[1] + 10) 							# Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (x_min, label_ymin-labelSize[1]-10), (x_min + labelSize[0], label_ymin + baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (x_min, label_ymin-7), font, 0.7, (0, 0, 0), txt_thickness) # Draw label text

#-------------------------------------------------------------------------------------------------------------------------------------------------

    # Draw FPS in corner of frame
    cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (30,50), font, font_size, (255,255,0), txt_thickness, cv2.LINE_AA)
    
    cv2.imshow('Object Detection', frame)

    t2 = cv2.getTickCount()		# End timer
    time1 = (t2-t1) / freq
    frame_rate_calc= 1 / time1		# Calculate framerate (FPS)

    # 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
videostream.stop()

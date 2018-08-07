#!/usr/bin/env python3
from mvnc import mvncapi as mvnc
import numpy as np
from ImageProcessor import ImageProcessor
import cv2
# name of the opencv window
cv_window_name = "MNIST Camera"
CAMERA_INDEX = 0
REQUEST_CAMERA_WIDTH = 640
REQUEST_CAMERA_HEIGHT = 480


# handles key presses
# raw_key is the return value from cv2.waitkey
# returns False if program should end, or True if should continue
def handle_keys(raw_key):
    global processor
    ascii_code = raw_key & 0xFF
    if ((ascii_code == ord('q')) or (ascii_code == ord('Q'))):
        return False
    elif (ascii_code == ord('w')):
        processor.p1 +=10
        print('processor.p1:' + str(processor.p1))
    elif (ascii_code == ord('s')):
        processor.p1 -=10
        print('processor.p1:' + str(processor.p1))
    elif (ascii_code == ord('a')):
        processor.p2 +=10
        print('processor.p2:' + str(processor.p2))
    elif (ascii_code == ord('d')):
        processor.p2 -=10
        print('processor.p1:' + str(processor.p2))
    return True
processor = ImageProcessor()
# get a list of names for all the devices plugged into the system
devices = mvnc.enumerate_devices()
if len(devices) == 0:
    print('No devices found')
    quit()

# get the first NCS device by its name.  For this program we will always open the first NCS device.
dev = mvnc.Device(devices[0])

# try to open the device.  this will throw an exception if someone else has it open already
try:
    dev.open()
except:
    print("Error - Could not open NCS device.")
    quit()

# Read a compiled network graph from file (set the graph_filepath correctly for your graph file)
with open("graph", mode='rb') as f:
    graphFileBuff = f.read()

graph = mvnc.Graph('graph1')

# Allocate the graph on the device and create input and output Fifos
in_fifo, out_fifo = graph.allocate_with_fifos(dev, graphFileBuff)
cv2.namedWindow(cv_window_name)
cv2.moveWindow(cv_window_name, 10,  10)

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUEST_CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUEST_CAMERA_HEIGHT)

actual_frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print ('actual video resolution: ' + str(actual_frame_width) + ' x ' + str(actual_frame_height))

if ((cap == None) or (not cap.isOpened())):
    print ('Could not open camera.  Make sure it is plugged in.')
    # print ('file name:' + input_video_file)
    print ('Also, if you installed python opencv via pip or pip3 you')
    print ('need to uninstall it and install from source with -D WITH_V4L=ON')
    print ('Use the provided script: install-opencv-from_source.sh')
    exit_app = True
    exit()
exit_app = False
while(True):
    ret, input_image = cap.read()

    if (not ret):
        print("No image from from video device, exiting")
        break

    # check if the window is visible, this means the user hasn't closed
    # the window via the X button
    prop_val = cv2.getWindowProperty(cv_window_name, cv2.WND_PROP_ASPECT_RATIO)
    if (prop_val < 0.0):
        exit_app = True
        break
    cropped_input, cropped = processor.preprocess_image(input_image)
    # Write the input to the input_fifo buffer and queue an inference in one call
    graph.queue_inference_with_fifo_elem(in_fifo, out_fifo, cropped_input.astype('float32'), 'user object')
    # Read the result to the output Fifo
    output, userobj = out_fifo.read_elem()
    predict_label = output.argmax()
    percentage = int(output[predict_label] * 100)
    label_text = str(predict_label) + " (" + str(percentage) + "%)"
    print('Predicted:',label_text)
    processor.postprocess_image(input_image, percentage, label_text, cropped)
    cv2.imshow(cv_window_name, input_image)
    raw_key = cv2.waitKey(1)
    if (raw_key != -1):
        if (handle_keys(raw_key) == False):
            exit_app = True
            break
cap.release()
# Deallocate and destroy the fifo and graph handles, close the device, and destroy the device handle
try:
    in_fifo.destroy()
    out_fifo.destroy()
    graph.destroy()
    dev.close()
    dev.destroy()
except:
    print("Error - could not close/destroy Graph/NCS device.")
    quit()
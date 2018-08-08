#!/usr/bin/env python3

# [NCSDK2 API](https://movidius.github.io/ncsdk/ncapi/ncapi2/py_api/readme.html)
from mvnc import mvncapi as mvnc
import numpy
import cv2
from ImageProcessor import ImageProcessor
processor = ImageProcessor()
test_image = './data/photo_6.jpg'
input_image = cv2.imread(test_image)
cropped_input, cropped = processor.preprocess_image(input_image)
# Using NCS Predict
# set the logging level for the NC API
# mvnc.global_set_option(mvnc.GlobalOption.RW_LOG_LEVEL, 0)

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

# Write the input to the input_fifo buffer and queue an inference in one call
graph.queue_inference_with_fifo_elem(in_fifo, out_fifo, cropped_input.astype('float32'), 'user object')

# Read the result to the output Fifo
output, userobj = out_fifo.read_elem()

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

print("NCS \r\n", output, '\r\nPredicted:',output.argmax())
#!/usr/bin/env python3

# [NCSDK2 API](https://movidius.github.io/ncsdk/ncapi/ncapi2/py_api/readme.html)
from mvnc import mvncapi as mvnc
# import mnist    # pip3 install mnist
from keras.datasets import mnist
import numpy

# Load MNIST data from Keras
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Load MNSIT data from mnist python package
# x_train = mnist.train_images()
# y_train = mnist.train_labels()

# x_test = mnist.test_images()
# y_test = mnist.test_labels()


# Prepare test image
test_idx = numpy.random.randint(0, 10000)
test_image = x_test[test_idx]
test_image = test_image.astype('float32') / 255.0

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
graph.queue_inference_with_fifo_elem(in_fifo, out_fifo, test_image, 'user object')

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
print("Correct:", y_test[test_idx])

# Intel® Movidius™ NCS MNIST example for NCSDK2

## Run MNIST classification model trained with Keras

## Quick start

* Plug NCS to a USB port on the host machine.
* Run command - `make all`
* Run command - `make run`
---
## Makefile
Provided Makefile describes various targets that help with the above mentioned tasks.

### make all
Runs Keras model training(only for the first time), ncprofile, nccheck, nccompile.

### make train
Train and save Keras MNIST model if Keras model files are not found.

### make profile
Runs the provided network on the NCS and generates per layer statistics that are helpful for understanding the performance of the network on the Neural Compute Stick.

### make compile
Uses the network description and the trained weights files to generate a Movidius internal 'graph' format file.  This file is later used for loading the network on to the Neural Compute Stick and executing the network.

### make run
Runs the provided predict-mnist-ncsdk2.py file which sends a single image to the Neural Compute Stick and receives and displays the inference results.

### make check
Runs the network on Caffe on CPU and runs the network on the Neural Compute Stick.  Check then compares the two results to make sure they are consistent with each other.

### make clean
Removes all the temporary files and trained model files that are created by the Makefile


---
After training the Keras MNIST model, 3 files will be generated, while the conversion script `convert-mnist.py` only use the first 2 files to generate TensorFlow model files into `TF_Model` directory.

model.json `Only contain model graph (Keras Format)`.

weights.h5 `Only contain model weights (Keras Format)`.

model.h5 `Both contain model graph and weights (Keras Format)`.


## Reference

+ [oraoto/learn_ml](https://github.com/oraoto/learn_ml/blob/master/ncs)
+ [ardamavi/Intel-Movidius-NCS-Keras](https://github.com/ardamavi/Intel-Movidius-NCS-Keras)

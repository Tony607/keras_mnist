#!/usr/bin/env python3

from keras import layers
from keras import models
from keras.models import load_model
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import backend as K
#import tensorflow as tf
import numpy

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Test image
test_idx = numpy.random.randint(0, 10000)
test_image = x_test[test_idx]
test_image = test_image.astype('float32') / 255.0

## With Keras
#model_file = "model.json"
#weights_file = "weights.h5"

#with open(model_file, "r") as file:
#    config = file.read()
#model = models.model_from_json(config)
#model.load_weights(weights_file)

model = load_model('model.h5')

output = model.predict(test_image.reshape(1, 28, 28, 1))[0]

print("Keras \r\n", output, '\r\nPredicted:',output.argmax())
print("Correct:", y_test[test_idx])
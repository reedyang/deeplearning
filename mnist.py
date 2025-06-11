import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

network = models.Sequential()

#network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Input(shape=(28 * 28,)))  # Specify the input shape
network.add(layers.Dense(512, activation='relu'))
#network.add(layers.Dense(512, activation='relu'))

#network.add(layers.Dense(10, activation='sigmoid'))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

print("Start trainging...")
network.fit(train_images, train_labels, epochs=5, batch_size=128)

print("\nEvaluating...")
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

# test with my own images
# scorecard for how well the network performs, initially empty
print("\nTest my own images...")
import glob
import numpy
import datetime
from PIL import Image

test_images = []
test_labels = []
# load the png image data as test data set
for image_file_name in glob.glob('my_own_images/2828_my_own*_?.png'):
    #print("loading ...", image_file_name)
    # using the filename to set the correct label
    label = int(image_file_name[-5:-4])

    image = Image.open(image_file_name).convert('L')
    width, height = image.size
    if width != 28 or height != 28:
        image = image.resize((28,28))
        pass
    
    # load image data from png files into an array
    img_array = numpy.asarray(image, dtype=float)
    img_data = 255.0 - img_array.reshape(784)
    #inputs = (img_data / 255.0 * 0.99) + 0.01  # (784,)
    inputs = img_data / 255.0  # (784,)

    '''
    predict_input = numpy.expand_dims(inputs, axis=0) # (1, 784)
    output = network.predict(predict_input) # (1, 10)
    print(numpy.argmax(output[0]))
    '''

    test_images.append(inputs)
    test_labels.append(label)

    pass

test_images = numpy.array(test_images) #(11, 784)
test_labels = numpy.array(test_labels) #(11,)
test_labels = to_categorical(test_labels) #(11, 10)

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)
"""
v00 - Forked from: brutally simple CNN with Keras, 98% Courtesy: https://www.kaggle.com/eiffelwong1

v03-04 trying some things from https://keras.io/examples/generative/vae/ I don't like the binary (no greyscale) image conversion, but interested in the convolutional front and back end...
"""

import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import tensorflow.keras as keras
import keras

#for data augmentation (to make more data from existing data by shifting, rotating, scaling, etc.)
# Data augmentation is NOT used in this version of the compressor
from keras.preprocessing.image import ImageDataGenerator

#simple CNN model with Keras
from keras.models import Model, Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization, Conv2DTranspose
from keras.layers import Dense, Flatten, Activation, Reshape

# For visualization
from matplotlib import pyplot

#reading both files
data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
val_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

NUM_CLASS = 10

#making one hot encoding for the label
label = data['label']
label_one_hot = np.zeros((label.size, NUM_CLASS))
for i in range(label.size):
    label_one_hot[i,label[i]] = 1
#remove the label column, so the remaining 784 columns can form a 28*28 photo
del data['label']

#changing data from DataFrame object to a numpy array, cause I know numpy better :p
# MADE THE DATA -127 to +128 instead of 0 to 255, to better match the residuals
# data = data.to_numpy()
data = data.to_numpy() - 127
print(data.shape)

#making data to 28*28 photo
data = data.reshape(-1,28,28,1)


#checking out data shape
print(' data shape: {} \n one hot lable shape: {}'.format(
    data.shape, label_one_hot.shape))
print(' data minimum: {} \n data maximim: {}'.format(
    data.min(), data.max()))

# generate samples and plot
for i in range(9):
        # define subplot
        pyplot.subplot(330 + 1 + i)
        # convert for viewing
        image = data[i][:,:,0]
        # plot raw pixel data
        pyplot.imshow(image, cmap="gray")
# show the figure
pyplot.show()

"""
The Nth Autoencoder
"""

# The Nth Autoencoder network model compresses the entire 28x28 pixel image down to a few floating point numbers
num_autoencoders = 3
# The number of floating point numbers images are compressed down to
compression = [1, 10, 100]
# number of training epochs per netowrk
num_epochs = 5


# the autoencoder models
model = []
# The front half (coder) models
c_model = []
# The decompressed approximations (decoded)
data_decomp = []
# The residual (difference between input and compressed appriximation)
data_res = []

for i in range (num_autoencoders) :
    this_compression = compression[i]
    model.append (Sequential([
        Convolution2D(filters = compression[i] * 4, kernel_size = (3,3), activation = 'relu', strides=2, padding="same", input_shape=(28,28,1), name='A'),
        Convolution2D(filters = compression[i] * 4, kernel_size = (3,3), activation = 'relu', strides=2, padding="same", name='B'),
        #Convolution2D(filters = compression[i] * 4, kernel_size = (3,3), activation = 'relu', strides=2, padding="same", name='C'),
        Flatten(name="flat"),
#        Dense(compression[i] * 2, activation='relu', name='D'),
        Dense(compression[i], activation='linear', name='code'),
        # decoding portion
        Dense(7 * 7 *  compression[i] * 4, activation='relu', name='D2'),
        Reshape((7, 7, compression[i] * 4), name='reshape'),
        #Conv2DTranspose(filters = compression[i] * 4, kernel_size = (3,3), activation = 'relu', strides=2, padding="same", name='E'),
        Conv2DTranspose(filters = compression[i] * 4, kernel_size = (3,3), activation = 'relu', strides=2,  padding="same", name='F'),
        Conv2DTranspose(compression[i] * 4, (3,3), activation="relu", strides=2, padding="same", name='G'),
        Conv2DTranspose(1, (3,3), activation="linear", padding="same", name='H')
        #  ---> End of Convolution

        #  ---> START Dense Only Compression (no knowledge that this is an image)
#        Flatten(input_shape=(28,28,1), name="flat"),
#        Dense(compression[i] * middle, activation='relu', name='middle1b'),
#        Dense(compression[i] * middle, activation='relu', name='middle1c'),
#        Dense(compression[i] * middle, activation='relu', name='middle1d'),
#        # Code Generation Layer
#        Dense(compression[i], activation='linear', name='code'),
#        Dense(compression[i] * middle, activation='relu', name='middle2b'),
#        Dense(compression[i] * middle, activation='relu', name='middle2c'),
#        Dense(compression[i] * middle, activation='relu', name='middle2d'),
#        Dense(784, activation='linear', name='decode'),
#        Reshape((28,28,1)),
        #  ---> END Dense Only        
        ]))
    model[i].compile('adam',
              loss='mse',
              metrics=['mse']
             )

    # Diplay the model summary
    print("model",i,"summary")
    model[i].summary()
    print ("\n")

    # Train the to recreate original image or residual
    if (i == 0) :
        model[i].fit(data, data, epochs = num_epochs, validation_split = 0.1)
    else :
        model[i].fit(data_res[i-1], data_res[i-1], epochs = num_epochs, validation_split = 0.1)

    # Save the model
    model[i].save('model_' + str(i))

    # The front half of the autoencoder is the "coder" part of this CODEC pair
    # We can use the coder to convert the image data (784 integers) into a much smaller compressed version
    # Load just the compression model
    c_model.append (Sequential([
        Convolution2D(compression[i] * 4, kernel_size = (3,3), activation = 'relu', strides=2, padding="same", input_shape=(28,28,1), name='A'),
        Convolution2D(filters = compression[i] * 4, kernel_size = (3,3), activation = 'relu',  strides=2, padding="same", name='B'),
        #Convolution2D(filters = compression[i] * 4, kernel_size = (3,3), activation = 'relu', strides=2, padding="same", name='C'),
        Flatten(name="flat"),
#        Dense(compression[i] * 2, activation='relu', name='D'),
        Dense(compression[i], activation='linear', name='code')
        #  ---> End of Convolution

        #  ---> START Dense Only Compression (no knowledge that this is an image)
#        Flatten(input_shape=(28,28,1), name="flat"),
#        Dense(compression[i] * middle, activation='relu', name='middle1b'),
#        Dense(compression[i] * middle, activation='relu', name='middle1c'),
#        Dense(compression[i] * middle, activation='relu', name='middle1d'),
#        # Code Generation Layer
#        Dense(compression[i], activation='linear', name='code'),
        #  ---> END Dense Only
        ]))
    c_model[i].load_weights('model_' + str(i), by_name=True)

    # Let's calculate the decompressed estimation and the residual (first residual minus decompressed approximation)
    if (i == 0) :
        data_decomp.append(model[i].predict(data))
        data_res.append(data - data_decomp[i])
        data_code = c_model[i].predict(data)
    else :
        data_decomp.append(model[i].predict(data_res[i-1]))
        data_res.append(data_res[i-1] - data_decomp[i])
        data_code = np.append(data_code, c_model[i].predict(data_res[i-1]), axis = 1)
        

from random import random
# Let's see a hadful of data samples, showing the original image followed by the compressed image
# generate samples and plot
num_draw = 3
for j in range (num_draw) :
    pick = int(random() * data.shape[0])
    for i in range(num_autoencoders + 1):
        # convert to np array for viewing
        if (i == 0) :
            image = data[pick][:,:,0]
        else :
            for k in range(i) :
                if k==0 :
                    image = data_decomp[k][pick][:,:,0]            
                else :
                    image += data_decomp[k][pick][:,:,0]            
        # define subplot
        pyplot.subplot(num_draw, num_autoencoders + 1, i + j * (num_autoencoders + 1) + 1)
        # plot raw pixel data
        pyplot.imshow(image, cmap='gray')
# show the figure
pyplot.show()


# Now, thanks to multiple layers of compression, the data is represented by a sequence of codes
data_code.shape

"""
Now to learn the digits using only the coded data
"""

num_feat = int(data_code[0].shape[0])

Fmodel = Sequential([
    Dense(num_feat * 2, activation='relu', input_shape=(num_feat,)),
    Dense(num_feat * 2, activation='relu'),
    Dense(10),
    Activation('softmax')
          ])

Fmodel.compile('adam',
              loss='categorical_crossentropy',
              metrics=['accuracy']
             )

# Diplay the model summary
print("Final model summary")
Fmodel.summary()
print ("\n")

history = Fmodel.fit(data_code, label_one_hot, epochs = 10, validation_split = 0.1)


import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()


#we read the csv before, but just read it again here.
val_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

#the same way to process the training data after seperating the label
val_data = val_data.to_numpy() - 127 # shifted the data to more resemble the residuals, which are both positive and negative values
val_data = val_data.reshape(-1,28,28,1)

# Encode the data
# The decompressed approximations (decoded)
val_decomp = []
# The residual (difference between input and compressed appriximation)
val_res = []
for i in range (num_autoencoders) :
    this_compression = compression[i]
    # Let's calculate the decompressed estimation and the residual (first residual minus decompressed approximation)
    if (i == 0) :
        val_decomp.append(model[i].predict(val_data))
        val_res.append(val_data - val_decomp[i])
        val_code = c_model[i].predict(val_data)
    else :
        val_decomp.append(model[i].predict(val_res[i-1]))
        val_res.append(val_res[i-1] - val_decomp[i])
        val_code = np.append(val_code, c_model[i].predict(val_res[i-1]), axis = 1)
       

    #here we ask the model to predict what the class is
raw_result = Fmodel.predict(val_code)

#note: model.predict will return the confidence level for all 10 class,
#      therefore we want to pick the most confident one and return it as the final prediction
result = np.argmax(raw_result, axis = 1)

#generating the output, remember to submit the result to the competition afterward for your final score.
submission = pd.DataFrame({'ImageId':range(1,len(val_data) + 1), 'Label':np.argmax(raw_result, axis = 1)})
submission.to_csv('SimpleCnnSubmission.csv', index=False)







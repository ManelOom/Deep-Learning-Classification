# Fix for the several warnings gotten when compiling Tensorflow, referring to the appropriate compiler flags.
# Fix found in: https://stackoverflow.com/questions/66092421/how-to-rebuild-tensorflow-with-the-compiler-flags

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy

# Firstly, we'll import our working Dataset. A Covid-19 Image Dataset deployed with the 
# goal of helping Deep Learning and AI Enthusiasts to contribute to improving COVID-19 detection using just Chest X-rays.
# The images were released by the University of Montreal.

# The data used in the project : https://www.kaggle.com/pranavraikokte/covid19-image-dataset
# This dataset is relatively small, containing 251 images in the 'train' directory and 66 images in the 'test' directory.

# This project's goals is to develop a learning model that supports doctors with diagnosing illnesses that affect patients’ lungs.

# Using the Keras module, we'll create a classification model that outputs a diagnosis based on a patient’s X-ray scan.
# Hoepfully this model can help doctors with the challenge of deciphering X-ray scans and open a dialogue
# between research teams and the medical staff to create learning models that are as effective and interpretable as possible.

# We're dealing with three different folders, one containing images of X-ray scans of people diagnosed with Covid-19, one containing
# images of X-ray scans of people diagnosed with viral pneumonia and another one with images of X-ray scans with images of people without any disease.
# Therefore, we are dealing with a multi-class classification problem.

# Firstly, we'll define some variables which describe the images' properties. The images are all 256x256 pixels and
# are in grayscale, so the maximum value a pixel can have is 255 (the minimum, of course, is 0).

Target_Size_Image = (256,256)
rescale_image = 1.0/255
COLOR_MODE = "grayscale"

# The most frequently tweaked model hyperparameters were batch size, learning rate, and
# the number of epochs to train over. We'll gather all of them into one convenient place to make
# it easier to change them during the optimization process.

BATCH_SIZE = 25
learning_rate_num = 0.003
num_of_epochs = 50

# Construct an ImageDataGenerator object with several data augmentation techniques for our training data. 
training_data_generator = ImageDataGenerator(rescale= rescale_image,

# Randomly flips images horizontally
horizontal_flip = True,

#Randomly increase or decrease the size of the image by up to 10%
zoom_range=0.1, 

#Randomly rotate the image between -25,25 degrees
rotation_range=25, 

#Shift the image along its width by up to +/- 5%
width_shift_range=0.05, 

#Shift the image along its height by up to +/- 5%
height_shift_range=0.05,

)

# Now we'll load and batch the training data into an iterator object.

training_iterator = training_data_generator.flow_from_directory("Covid19-dataset/train",class_mode='categorical',color_mode='grayscale', target_size = Target_Size_Image,batch_size=BATCH_SIZE)

# The following piece of code will print out the batch shape and label shape.

train_batch_input, train_batch_labels = training_iterator.next()
print('----------------------------------------------')
print(train_batch_input.shape, train_batch_labels.shape)
print('----------------------------------------------')

# Load the validation data using an ImageDataGenerator() object.

print("\nLoading validation data...")

# Because our validation data will be loaded using this object, no data augmentation techniques are specified.
# But we'll still scale our data, making sure all our values lie between 0 and 1.

validation_data_generator = ImageDataGenerator(rescale = rescale_image)

# Load and batch the validation data into an iterator object.

validation_iterator = validation_data_generator.flow_from_directory("Covid19-dataset/test",class_mode='categorical', color_mode='grayscale', target_size = Target_Size_Image, batch_size=BATCH_SIZE)

# The following piece of code will print out the batch shape and label shape

validation_batch_input, validation_batch_labels = validation_iterator.next()
print('----------------------------------------------')
print(validation_batch_input.shape, validation_batch_labels.shape)
print('----------------------------------------------')

print("\nBuilding model...")

def design_model(training_data):
    # Initiate a Sequential model
    model = Sequential()
    # Add Input layer with grayscale image shape
    model.add(tf.keras.Input(shape=(256, 256, 1)))
    # Add a Conv2D() layer featuring 10 5x5 filters with a stride of 3, with the default padding, using a relu activation function
    model.add(layers.Conv2D(10, 5, strides=3, activation="relu")) 
    # Add a MaxPooling2D() layer with a 3x3 window size and a stride of 3 
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(3,3)))
    # Add a Conv2D() layer featuring 10 4x4 filters with a stride of 2, with the default padding, using a relu activation function
    model.add(layers.Conv2D(10, 4, strides=2, activation="relu")) 
    # Add a MaxPooling2D() layer with a 2x2 window size and a stride of 2
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    # Add a Conv2D() layer featuring 8 3x3 filters with a stride of 1, with the default padding, using a relu activation function
    model.add(layers.Conv2D(8, 3, strides=1, activation="relu"))
    # Add a MaxPooling2D() layer with a 2x2 window size and a stride of 2
    #model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
     # Add a Conv2D() layer featuring 6 2x2 filters with a stride of 1, with the default padding, using a relu activation function
    #model.add(layers.Conv2D(6, 2, strides=1,padding = 'same', activation="relu"))
    # Add a Flatten() layer to translate the input images into a single vector
    model.add(layers.Flatten())
    # Finally, we'll add the output layer, a Dense() layer with 3 perceptrons and a softmax activation function
    model.add(layers.Dense(3,activation="softmax"))
    
    # Now we'll choose an optimization algorithm, with its learning rate specified, Adam was the chosen optimization algorithm.
    # We'll compile the model using categorical crossentropy as the loss function, choosing categorical accuracy and AUC
    # as the metrics to be tracked as the model is trained.
    print("\nCompiling model...")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_num), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.AUC()],)
    
    # summarize model
    model.summary()
    return model

# Keeping the prevention of overfitting in mind, we'll instantiate an EarlyStopping() object. This object will stop the model training
# if the validation loss, which in our case is categorical cross-entropy, plateaus or increases after reaching a minimum value.
# We'll monitor the validation loss, making sure that once early stopping has been triggered, we'll keep training for 50 epochs to see if
# a plateau has started to increase or decrease.

earlystop_model = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 50)
    

# use model function
model = design_model(training_iterator)

print("\nTraining model...")

# fit the model with 10 ephochs and early stopping
history =model.fit(
        training_iterator,
        steps_per_epoch=training_iterator.samples/BATCH_SIZE, epochs=num_of_epochs,
        validation_data=validation_iterator,
        validation_steps=validation_iterator.samples/BATCH_SIZE,
        callbacks=[earlystop_model])


# plotting categorical and validation accuracy over epochs
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['categorical_accuracy'])
ax1.plot(history.history['val_categorical_accuracy'])
ax1.set_title('model accuracy')
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
ax1.legend(['train', 'validation'], loc='upper left')

# plotting auc and validation auc over epochs
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['auc'])
ax2.plot(history.history['val_auc'])
ax2.set_title('model auc')
ax2.set_xlabel('epoch')
ax2.set_ylabel('auc')
ax2.legend(['train', 'validation'], loc='upper left')

plt.show()


predictions = model.predict(validation_iterator, steps=validation_iterator.samples / BATCH_SIZE)
predicted_classes = numpy.argmax(predictions, axis=1)
true_classes = validation_iterator.classes
class_labels = list(validation_iterator.class_indices.keys())
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)   

cm=confusion_matrix(true_classes,predicted_classes)
print(cm)
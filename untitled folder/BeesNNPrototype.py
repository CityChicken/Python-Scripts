# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 22:54:19 2018

@author: Jacob
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import time
import theano
import theano.tensor as T
from PIL import Image
from lasagne import layers
from lasagne import objectives
from lasagne import updates
from lasagne.nonlinearities import softmax, tanh


def buildModel():
 
    #this is our input layer with the inputs (None, dimensions, width, height)
    l_input = layers.InputLayer((None, 3, 90, 90))
 
    #first convolutional layer, has l_input layer as incoming and is followed by a pooling layer
    l_conv1 = layers.Conv2DLayer(l_input, num_filters=45, filter_size=3, pad='same', nonlinearity=tanh)
    l_pool1 = layers.MaxPool2DLayer(l_conv1, pool_size=2)
 
    #second convolution (l_pool1 is incoming), let's increase the number of filters
    l_conv2 = layers.Conv2DLayer(l_pool1, num_filters=64, filter_size=3, pad='same', nonlinearity=tanh)
    l_pool2 = layers.MaxPool2DLayer(l_conv2, pool_size=3)
 
    #third convolution (l_pool2 is incoming), even more filters
    l_conv3 = layers.Conv2DLayer(l_pool2, num_filters=128, filter_size=3, pad='same', nonlinearity=tanh)
    l_pool3 = layers.MaxPool2DLayer(l_conv3, pool_size=3)
 
    #our cnn contains 3 dense layers, one of them is our output layer
    l_dense1 = layers.DenseLayer(l_pool3, num_units=128, nonlinearity=tanh)
    l_dense2 = layers.DenseLayer(l_dense1, num_units=128, nonlinearity=tanh)
 
    #the output layer has 5 units which is exactly the count of our class labels
    #it has a softmax activation function, its values represent class probabilities
    l_output = layers.DenseLayer(l_dense2, num_units=7, nonlinearity=softmax)
 
    #let's see how many params our net has
    print("MODEL HAS", layers.count_params(l_output), "PARAMS")
    #we return the layer stack as our network by returning the last layer
    return l_output
 
NET = buildModel()

#################### LOSS FUNCTION ######################
def calc_loss(prediction, targets):
 
    #categorical crossentropy is the best choice for a multi-class softmax output
    l = T.mean(objectives.categorical_crossentropy(prediction, targets))
    
    return l
 
#theano variable for the class targets
#this is the output vector the net should predict
targets = T.matrix('targets', dtype=theano.config.floatX)
 
#get the network output
prediction = layers.get_output(NET)
 
#calculate the loss
loss = calc_loss(prediction, targets)

################# ACCURACY FUNCTION #####################
def calc_accuracy(prediction, targets):
    targets = theano.tensor.argmax(targets, axis=-1)
    #we can use the lasagne objective categorical_accuracy to determine the top1 accuracy
    top = theano.tensor.argmax(prediction, axis=-1)
    a = theano.tensor.eq(top, targets)
    #a = T.mean(T.eq(T.argmax(prediction, axis=0), targets))
    return a
 
accuracy = calc_accuracy(prediction, targets)

####################### UPDATES #########################
#get all trainable parameters (weights) of our net
params = layers.get_all_params(NET, trainable=True)
 
#we use the adam update
#it changes params based on our loss function with the learning rate
param_updates = updates.adam(loss, params, learning_rate=0.0001)

#the theano train functions takes images and class targets as input
#it updates the parameters of the net and returns the current loss as float value
#compiling theano functions may take a while, you might want to get a coffee now...
print("COMPILING THEANO TRAIN FUNCTION...")
train_net = theano.function([layers.get_all_layers(NET)[0].input_var, targets], loss, updates=param_updates)
print("DONE!")
 
################# PREDICTION FUNCTION ####################
#we need the prediction function to calculate the validation accuracy
#this way we can test the net after training
#first we need to get the net output
net_output = layers.get_output(NET)
 
#now we compile another theano function; this may take a while, too
print("COMPILING THEANO TEST FUNCTION...")
test_net = theano.function([layers.get_all_layers(NET)[0].input_var, targets], [net_output, loss, accuracy]) 
print("DONE!")




masterpath = "/Users/CupulFamily/Documents/PythonProjects/CNN/"
loadpath = "bee_imgs_alt/"
imagepath = "bee_imgs/"
beemasterkey = "bee_data.csv"
beeimg = "001_043.png"


image = mpimg.imread(masterpath + imagepath + beeimg)
imgplot = plt.imshow(image)
plt.show()

beemasterkey_df = pd.read_csv(masterpath + beemasterkey)
print(beemasterkey_df.head(5))

#print(beemasterkey_df['file'])
k = len(beemasterkey_df.index)
dimensions = np.zeros((k,3))
desired_size = 90
Y = np.zeros((k,7))
for i in range (0,k):
# =============================================================================
#     image = Image.open(masterpath + imagepath 
#                          + beemasterkey_df.iloc[i]['file'])
#     #print(image.shape)
#     dimensions[i,0:2] = image.size
#     old_size = ((dimensions[i,0],dimensions[i,1]))
# =============================================================================
    #print(old_size)
    if beemasterkey_df.iloc[i]['subspecies'] == '-1':
        Y[i] = [1.0,0.0,0.0,0.0,0.0,0.0,0.0]
    elif beemasterkey_df.iloc[i]['subspecies'] == 'Italian honey bee':
        Y[i] = [0.0,1.0,0.0,0.0,0.0,0.0,0.0]
    elif beemasterkey_df.iloc[i]['subspecies'] == 'VSH Italian honey bee':
        Y[i] = [0.0,0.0,1.0,0.0,0.0,0.0,0.0]
    elif beemasterkey_df.iloc[i]['subspecies'] == 'Carniolan honey bee':
        Y[i] = [0.0,0.0,0.0,1.0,0.0,0.0,0.0]
    elif beemasterkey_df.iloc[i]['subspecies'] == 'Russian honey bee':
        Y[i] = [0.0,0.0,0.0,0.0,1.0,0.0,0.0]
    elif beemasterkey_df.iloc[i]['subspecies'] == '1 Mixed local stock 2':
        Y[i] = [0.0,0.0,0.0,0.0,0.0,1.0,0.0]
    elif beemasterkey_df.iloc[i]['subspecies'] == 'Western honey bee':
        Y[i] = [0.0,0.0,0.0,0.0,0.0,0.0,1.0]    
    
# =============================================================================
#     if i == 28:
#         imgplot = plt.imshow(image)
#         plt.show()
#     
#         ratio = float(desired_size)/max(old_size)
#         new_size = tuple([int(x*ratio) for x in old_size])
#         print(new_size)
#         image = image.resize(new_size, Image.ANTIALIAS)
#         new_im = Image.new("RGB", (desired_size, desired_size), (0,0,0))
#         new_im.paste(image, ((desired_size-new_size[0])//2,
#                         (desired_size-new_size[1])//2))
#         
#         imgplot = plt.imshow(new_im)
#         plt.show()
#     
#     ratio = float(desired_size)/max(old_size)
#     new_size = tuple([int(x*ratio) for x in old_size])
#     print(new_size)
#     image = image.resize(new_size, Image.ANTIALIAS)
#     new_im = Image.new("RGB", (desired_size, desired_size), (0,0,0))
#     new_im.paste(image, ((desired_size-new_size[0])//2,
#                     (desired_size-new_size[1])//2))
#     new_im.save(masterpath + loadpath 
#                          + beemasterkey_df.iloc[i]['file'])
# =============================================================================
batch_size = 128

test_proportion = 0.8167
num_train = round(test_proportion * k)
num_test = k - num_train
rindex = np.random.permutation(k)
num_subsets = int(np.ceil(num_train / batch_size))+1
z = 0
i = 0
subset_indices = np.zeros((num_subsets,1))
while z < num_train:
    z= z+batch_size
    i = i+1
    if z > num_train:
        subset_indices[i] = num_train
    else:
        subset_indices[i] = z

start_i = subset_indices[0]
end_i = subset_indices[1] - 1

    
x_b = np.zeros((batch_size,3,desired_size,desired_size))
y_b = np.zeros((batch_size, 7))

x_test = np.zeros((num_test,3,desired_size,desired_size))
y_test = np.zeros((num_test, 7))

for i in range(int(subset_indices[-1]),k-1):
    image = Image.open(masterpath + loadpath 
                          + beemasterkey_df.iloc[rindex[i]]['file'])
    image.load()  
    x_test[i - int(subset_indices[-1])] = np.transpose(np.asarray(image, dtype="int32"),(2,0,1) )
    y_test[i - int(subset_indices[-1])] = Y[rindex[i]]


##################### STAT PLOT #########################
plt.ion()
def showChart(epoch, t, v, a, path):

    #new figure
    plt.figure(0)
    plt.clf()

    #x-Axis = epoch
    e = range(0, epoch)

    #loss subplot
    plt.subplot(211)
    plt.plot(e, train_loss, 'r-', label='Train Loss')
    plt.plot(e, val_loss, 'b-', label='Val Loss')
    plt.ylabel('loss')

    #show labels
    plt.legend(loc='upper right', shadow=True)

    #accuracy subplot
    plt.subplot(212)
    plt.plot(e, val_accuracy, 'g-')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')

    #show
    plt.show()
    plt.savefig(path + 'cnn_results.png')
    plt.pause(0.5)

    
#print(subset_indices)
#print(k,num_train,num_test)
#for i in range(0,batch_size):
    

###################### TRAINING #########################
print("START TRAINING...")
train_loss = []
val_loss = []
val_accuracy = []
for epoch in range(1, 31):
    #start timer
    start = time.time()
 
    #iterate over train split batches and calculate mean loss for epoch
    t_l = []
    for j in range(len(subset_indices) - 1):
        #print(int(subset_indices[j]),int(subset_indices[j+1]))
        for i in range(int(subset_indices[j]),int(subset_indices[j+1])):
            image = Image.open(masterpath + loadpath 
                                  + beemasterkey_df.iloc[rindex[i]]['file'])
            image.load()  
            x_b[i - int(subset_indices[j])] = np.transpose(np.asarray(image, dtype="int32"),(2,0,1) )
            y_b[i - int(subset_indices[j])] = Y[rindex[i]]
    
    
            #calling the training functions returns the current loss
        l = train_net(x_b, y_b)
        t_l.append(l)
    #we validate our net every epoch and pass our validation split through as well
    v_l = []
    v_a = []

    #calling the test function returns the net output, loss and accuracy
    prediction_batch, l2, a2 = test_net(x_test, y_test)
    v_l.append(l2)
    v_a.append(a2)
 
    #stop timer
    end = time.time()
 
    #calculate stats for epoch
    train_loss.append(np.mean(t_l))
    val_loss.append(np.mean(v_l))
    val_accuracy.append(np.mean(v_a))
 
    #print stats for epoch
    print("EPOCH:", epoch,
        "TRAIN LOSS:", train_loss[-1],
        "VAL LOSS:", val_loss[-1],
        "VAL ACCURACY:", (int(val_accuracy[-1] * 1000) / 10.0), "%",
        "TIME:", (int((end - start) * 10) / 10.0), "s")
 
    #show chart
    showChart(epoch, train_loss, val_loss, val_accuracy, masterpath)
 
print("TRAINING DONE!")

















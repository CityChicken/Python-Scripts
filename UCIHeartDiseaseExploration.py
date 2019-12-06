# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 22:47:21 2019

@author: Jacob
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn import svm
#import tensorflow as tf
masterpath = "F:/MachineLearningData/HeartDiseaseUCI/"
masterkey = "heart.csv"

data_set = pd.read_csv(masterpath + masterkey)
print(data_set.head())
names = list(data_set.columns.values)
#data_set.hist()
#data_set.plot(kind='density', subplots=True, layout=(4,4), sharex=False)
#data_set.plot(kind='box', subplots=True, layout=(4,4), sharex=False)
correlations = data_set.corr()

#fig = plt.figure()
#ax = fig.add_subplot(111)
#cax = ax.matshow(correlations, vmin = -1, vmax = 1)
#fig.colorbar(cax)
#ticks= np.arange(0,14,1)
#ax.set_xticks(ticks)
#ax.set_yticks(ticks)
#ax.set_xticklabels(names)
#ax.set_yticklabels(names)
#mng = plt.get_current_fig_manager()
#mng.window.showMaximized()

#pd.scatter_matrix(data_set)
#mng = plt.get_current_fig_manager()
#mng.window.showMaximized()
#plt.show()


attributes = data_set.iloc[:,:12]
labels = data_set.iloc[:,13]

def split_data(attribute_frame, label_frame, x):
    if x<= 0 or x>=1:
        raise Exception('The value of x must be between 0 and 1. The value of x was' + x)
    else:
        dimensions = attribute_frame.shape
        upper_limit = int( dimensions[0] * x)
        subset = np.random.permutation(np.arange(0,dimensions[0]-1,1))
        training_set = attribute_frame.iloc[subset[:upper_limit-1],:]
        validation_set = attribute_frame.iloc[subset[upper_limit:],:]
        training_lables = label_frame.iloc[subset[:upper_limit-1]]
        validation_lables = label_frame.iloc[subset[upper_limit:]]
        
    return training_set, validation_set, training_lables, validation_lables;
 
avg_accuracy = 0     
  
for i in np.arange(0,10):
    training_attr, validation_attr, training_lbl, validation_lbl  = split_data(attributes, labels, 0.90)
    model1 = svm.LinearSVC()
    model1.fit(training_attr, training_lbl)
    avg_accuracy = model1.score(validation_attr, validation_lbl) + avg_accuracy

avg_accuracy = avg_accuracy / 10

print(avg_accuracy)
        





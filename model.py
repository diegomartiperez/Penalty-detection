from sre_compile import isstring
from traceback import format_exception_only
from numpy import loadtxt
import os
import numpy as np
import cv2
from glob import glob
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import tensorflow as tf
#from scikits.statsmodels.tools import categorical
import torch
import random
from load_data import * 
from display_video import * 
import matplotlib.pyplot as plt
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.optimizers import Adam
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from keras.utils import to_categorical
from matplotlib import pyplot
from keras import Model
from keras.layers import Input, Dense, Bidirectional
import shap
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.inspection import permutation_importance
from matplotlib import pyplot
videos_folder = glob("runs/detect/test2/*")
annotations_folder = glob("json/test2/*")
labels_folder = glob("labels/test2/*")
labels_prefix = "labels/test2/"

videos_folder2 = glob("runs/detect/test/*")
annotations_folder2 = glob("json/test/*")
labels_folder2 = glob("labels/test/*")
labels_prefix2 = "labels/test/"

videos_folder3 = glob("runs/detect/all_vids/*")
annotations_folder3 = glob("json/all_vids/*")
labels_folder3 = glob("labels/all_vids/*")
labels_prefix3 = "labels/all_vids/"

frames = 10
#data_test2 = get_data(videos_folder,annotations_folder,labels_folder,labels_prefix)
#data_test,labels_test = get_data(videos_folder2,annotations_folder2,labels_folder2,labels_prefix2,frames)
#new_data,labels_train = get_data(videos_folder3,annotations_folder3,labels_folder3,labels_prefix3,frames)

import csv
"""
with open("train.csv", "w",newline='') as f:
    wr = csv.writer(f)
    wr.writerows(new_data)

with open("test.csv", "w",newline='') as f:
    wr = csv.writer(f)
    wr.writerows(data_test)
with open("train_labels.csv", "w",newline='') as f:
    wr = csv.writer(f)
    wr.writerows(labels_train)
with open("test_labels.csv", "w",newline='') as f:
    wr = csv.writer(f)
    wr.writerows(labels_test)
"""  
with open('data/train.csv', 'r') as read_obj:  
    train_data_csv = [[float(x) for x in rec] for rec in csv.reader(read_obj, delimiter=',')]
with open('data/test.csv', 'r') as read_obj:  
    test_data_csv = [[float(x) for x in rec] for rec in csv.reader(read_obj, delimiter=',')]
with open('data/train_labels.csv', newline='') as f:
    reader = csv.reader(f)
    train_labels = list(reader)
with open('data/test_labels.csv', newline='') as f:
    reader = csv.reader(f)
    test_labels = list(reader)

def list_of_lists(list):  
    list = [list[i:i+5] for i in range(0, len(list), 5)]
    return list

train_data = []
test_data = []
for i in range(len(train_data_csv)):
    new = list_of_lists(train_data_csv[i])
    train_data = train_data + [new]
for i in range(len(test_data_csv)):
    new = list_of_lists(test_data_csv[i])
    test_data = test_data + [new]

def prepare_data(data,labels):
    length = 0
    for listt in data:
        if len(listt)>length:
            length = len(listt)
    for listt in data:
        if len(listt)<length:
            for i in range(length-len(listt)):
                listt.append([0,0,0,0,0])
               # list.append([0,0,0,0])
    c = list(zip(data,labels))
    random.shuffle(c)
    data,labels = zip(*c)
    #random.shuffle(data)
    names = [item[0] for item in labels]
    labels = [item[1] for item in labels]   
    labels_array = np.array(labels) 
    no_action = 0
    penalty = 0
    for i in range(len(labels_array)):
        if labels_array[i]=="no-action":
            labels_array[i] = 0
            no_action = no_action + 1
        elif labels_array[i]=="penalty-action":
            labels_array[i] = 1
            penalty = penalty + 1
    labels_array = np.asarray(labels_array).astype("float64")
   # for item in data:
    #    del item[:2]
    data_array = np.array([np.array(item) for item in data])
    return data_array, labels_array,names

def prepare_test(data,labels):
    length = 0
    for list in data:
        if len(list)>length:
            length = len(list)
    for list in data:
        if len(list)<length:
            for i in range(length-len(list)):
                list.append([0,0,0,0,0])
                #list.append([0,0,0,0])
    names = [item[0] for item in labels]
    labels = [item[1] for item in labels]
    
    labels_array = np.array(labels) 
    no_action = 0
    penalty = 0
    for i in range(len(labels_array)):
        #print(labels_array[i])
        if labels_array[i]=="no-action":
            labels_array[i] = 0
            no_action = no_action + 1
        elif labels_array[i]=="penalty-action":
            labels_array[i] = 1
            penalty = penalty + 1
    labels_array = np.asarray(labels_array).astype("float64")
  #  for item in data:       
   #     del item[:2]   
    data_array = np.array([np.array(item) for item in data])
    return data_array, labels_array,names

def get_size(data,size):
    for list in data:
            if len(list)<size:
                for i in range(size-len(list)):
                    list.append([0,0,0,0,0])
    return np.array(data)

x_train, y_train,names= prepare_data(train_data, train_labels)
x_test, y_test,names_test = prepare_test(test_data, test_labels)
x_test = get_size(x_test.tolist(),x_train.shape[1])
print(names)
print(names_test)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

nopen_train = 0
pen_train = 0
nopen_test = 0
pen_test = 0
for i in range(len(y_train)):
    if (y_train[i] == 0):
        nopen_train = nopen_train + 1
    elif (y_train[i] == 1):
        pen_train = pen_train + 1
for i in range(len(y_test)):
    if (y_test[i] == 0):
        nopen_test = nopen_test + 1
    elif (y_test[i] == 1):
        pen_test = pen_test + 1
print("No penalties for train:",nopen_train)
print("Penalties for train:",pen_train)
print("No penalties for test:",nopen_test)
print("Penalties for test:",pen_test)

# define model for simple BI-LSTM + DNN based binary classifier
def define_model(data):
    input_shape = data.shape[1],data.shape[2]
    input1 = Input(shape=input_shape) #use row and column size as input size
    lstm1 = Bidirectional(LSTM(units=32))(input1)
    dnn_hidden_layer1 = Dense(3, activation='relu')(lstm1)
    dnn_output = Dense(1, activation='sigmoid')(dnn_hidden_layer1)
    model = Model(inputs=[input1],outputs=[dnn_output])
    # compile the model
    #opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    model.summary()
    return model
#model = define_model(data_array)
model = define_model(x_train)
class_weight = {0: 1,
                1: 2}
#earlyStopCallBack = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
history = model.fit(x_train,y_train,class_weight=class_weight,validation_split=0.2,epochs=300,verbose=1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.show()
pred = model.predict(x_test),y_test

def get_pred(pred,threshold):
    a = [0] * len(pred[0])
    y_pred = np.array(a)
    for i in range(len(pred[0])):
           # print(y_pred[i])
            if pred[0][i] < threshold:
                y_pred[i] = 0
                #print(pred[0][i],pred[1][i])
            elif pred[0][i] > threshold:
                y_pred[i] = 1
                #print(pred[0][i],pred[1][i])
    print(y_pred,pred[1])
    return y_pred
def get_confusion_matrix(pred,threshold):
    pos = 0
    neg = 0
    true_pos = 0 
    true_neg = 0
    false_pos = 0 
    false_neg = 0
    for i in range(len(pred[0])):
       # print(pred[0][i],pred[1][i])
        if (pred[0][i]>threshold and pred[1][i] == 1): #or (pred[0][i]<0.5 and pred[1][i] == 0):
            true_pos = true_pos + 1
        elif pred[0][i]<threshold and pred[1][i] == 0:
            true_neg = true_neg + 1
        elif pred[0][i]>threshold and pred[1][i] == 0:
            false_pos = false_pos + 1
        elif pred[0][i]<threshold and pred[1][i] == 1:
            false_neg = false_neg + 1
    correct = true_pos + true_neg
    incorrect = false_pos + false_neg
    for i in range(len(pred[0])):
        if pred[1][i] == 1:
            pos = pos +1
        elif pred[1][i] == 0:
            neg = neg + 1
    print("P:",pos,"N:",neg)
    print("TP:",true_pos,"TN:",true_neg,"FN:",false_neg,"FP:",false_pos)
    print("right:",correct,"wrong:",incorrect)
    accuracy = correct/(correct+incorrect)
    print("Test accuracy:", accuracy)
    y_pred = get_pred(pred,threshold)
    result = confusion_matrix(y_test, y_pred)
    labelsss = ["No penalty","Penalty"]
    disp = ConfusionMatrixDisplay(confusion_matrix=result, display_labels=labelsss)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()
    height = [true_pos,true_neg,false_pos,false_neg]
    bars = ('TP', 'TN', 'FP', 'FN')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()
    return y_pred, true_pos,true_neg,false_pos,false_neg

#d,true_pos,true_neg,false_pos,false_neg = get_confusion_matrix(pred,threshold=0.4)
#get_confusion_matrix(pred,threshold=0.45)
#get_confusion_matrix(pred,threshold=0.5)
#get_confusion_matrix(pred,threshold=0.55)
#i,true_pos3,true_neg3,false_pos3,false_neg3 = get_confusion_matrix(pred,threshold=0.6)

y_pred,true_pos2,true_neg2,false_pos2,false_neg2 = get_confusion_matrix(pred,threshold=0.5)
"""
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred)
auc_keras = auc(fpr_keras, tpr_keras)

plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
"""
#X = ['TP','TN','FP','FN']
#t1 = [true_pos,true_neg,false_pos,false_neg]
#t3 = [true_pos3,true_neg3,false_pos3,false_neg3]
#t2 = [true_pos2,true_neg2,false_pos2,false_neg2]
  
#X_axis = np.arange(len(X))
  
#plt.bar(X_axis - 0.20, t1, 0.2, label = 'Threshold = 0.4')
#plt.bar(X_axis + 0.0, t2, 0.2, label = 'Threshold = 0.5')
#plt.bar(X_axis + 0.20, t3, 0.2, label = 'Threshold = 0.6')
  
#plt.xticks(X_axis, X)
#plt.xlabel("Groups")
#plt.ylabel("Number of Students")
#plt.title("Comparison of different decision thresholds")
#plt.legend()
#plt.show()

def output_video(videos_folder,frames):
    for video in videos_folder:     
        count = 0    
        for ele in names_test:
            print(ele)
            if (ele == video.split("\\")[1].split(".")[0]):
                if count == 0:
                    index = names_test.index(ele)
                    count = count + 1
                else:
                    count = count + 1  
        #print(video.split("\\")[1].split(".")[0],count,index)
        play_video(names_test, video,y_pred,y_test,index,frames)
 
#videos_folder = glob("videos/*")
#output_video(videos_folder,frames)



from traceback import format_exception_only
from numpy import loadtxt
import os
import numpy as np
import cv2
from glob import glob
import pandas as pd
from read_json import * 

def get_labels_name(labels):
    name_txt = labels.split("\\")[1]
    if "gopro" in name_txt:
        if "gopro_post_goal_1" in name_txt:
            split = "_" 
            temp = name_txt.split(split)
            res = split.join(temp[:4]), split.join(temp[4:])
            number = res[1].split(".")[0]
            name = res[0]
        else:
            split = "_" 
            temp = name_txt.split(split)
            res = split.join(temp[:3]), split.join(temp[3:])
            number = res[1].split(".")[0]
            name = res[0]
    else:
        name = name_txt.split("_")[0]
        number = name_txt.split("_")[1].split(".")[0]
    return name,number

def get_txt(video,folder_labels,prefix):
    video_name = video.split("\\")[1].split(".")[0]
    count = 0
    list = []
    #print(folder_labels)
    for labels in folder_labels: 
        name,number = get_labels_name(labels)  
        if name == video_name:
            count = count+1
            label = prefix +"\\"+name+"_"+str(number)+".txt"
            list.append(label)
    return list
    
   #output list of lists of groups of frames
def join_frames(list,x):
    list = [list[i:i+x] for i in range(0, len(list), x)]
    return(list)

def get_order(list,x):
    nums =[]
    list2=[None]*len(list)
    for item in list:
        name,number = get_labels_name(item)
        num = int(number)
        list2[num-1] = item
    return join_frames(list2,x)

def read(file,groups): #read all the lines in a file, return a list of [object_id,x,y,w,h]
    with open(file) as f:
        array = []
        count_goal = 0
        count_ball = 0
        for line in f: # read rest of lines 
            #if count_goal == 0 or count_ball == 0:
            line = line.split(" ")  #football, goal, goal-bottom-corner, goal-top-corner, goalkeeper, football-player 
            #and count_ball == 0 
            """
            if line[0]=="6" and count_ball == 0:  #or line[0]=="7" or line[0]=="8" or line[0]=="9" or line[0]=="10" or line[0]=="11":
                #print(line)
                count_ball = 1
                array.append([float(x) for x in line if x!="6"])
              """
            """
            
            elif line[0]=="8" and count_goal == 0:
                array.append([float(x) for x in line])
                count_goal = 1                 
            else:
                pass
            """
            if line[0]=="6" or line[0]=="7" or line[0]=="8" or line[0]=="9" or line[0]=="10" or line[0]=="11":
                #groups.append([float(x) for x in line])
                for x in line:
                    groups.append(float(x))
                #del array[0]
                #print(array)
    #return array
    return groups

def get_dataframe(labels,group_of_frames,folder_annotations):    
    data = []
    new= [] 
    count = 0
    count_files = 0
    annotations = annotations_dict(folder_annotations)
    #labels will be list of lists of frames
    for group in labels:  
        name,number = get_labels_name(group[0])
        num = int(number) #frame
        if num>=annotations[(name)][1] and num<=annotations[(name)][2]:
            label = annotations[(name)][0]
        elif num>=annotations[(name)][4] and num<=annotations[(name)][5]:
            label = annotations[(name)][3]
        else:
            label = annotations[(name)][6]
        groups = []
        groups = [name,label]
        count = count + group_of_frames
        for item in group: 
            count_files = count_files+1  #read each frame file in a group of frames     
            read(item,groups) 
        data =  data + [groups]    #add group of frames with the rest
        #add annotation
    print("number of labels for 1 group:",len(data[0])-1)
    print("number of frames per group:",group_of_frames)
    print("num of bounding box files:",count_files)
    print("Number of samples (groups of frames):",len(data))
    return data

def get_data(videofolder,annotationfolder,labelfolder,prefix,group_of_frames):
    labels =[]
    #group_of_frames = 1
    for video in videofolder:
        #print(video)
        new = get_txt(video,labelfolder,prefix)
        new = get_order(new,group_of_frames)
        labels = labels + new
    #print(labels)
    #print(videofolder,annotationfolder,labelfolder)
    data = get_dataframe(labels,group_of_frames,annotationfolder)
    labels = [item[1] for item in data]  
    names = [item[0] for item in data] 
    labels_names = []
    for i in range(len(labels)):
        labels_names.append([names[i],labels[i]])
    for item in data:
        del item[:2]
    return data,labels_names





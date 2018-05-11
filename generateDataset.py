#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 14:29:38 2018

@author: prashant
"""
'''
Here we will read the preprocessed image and create array X and Y.
And this X and Y will be used as input to Neural Network.
'''
import Preprocess as pp
import os
import cv2
import numpy as np

RAW = '/home/prashant/Downloads/OdiaCharacterRecognition/Code/Data/images/alphaNumeric1.png'

def encode_labels(Y, num_labels):
    onehot = np.zeros((num_labels, Y.shape[0]))
    m = Y.shape[0] #no. of training example
    for i in range(m):
        onehot[Y[i], i] = 1
    return onehot
        
    
def generateDataset(folder):
    '''
    This function takes the images and store them as a 
    matrix form.All images will be stored in X and corresponding
    label will be stored in Y.
    '''
    X = []
    Y = []
    for file in os.listdir(folder):
        path = os.path.join(folder,file)
        if os.path.isdir(path) == True:
            generateDataset(path)
        else:
            #read only jpg file
            name = file.split('.')
            if name[-1] == 'jpg':
                #read image file
                img = cv2.imread(path, cv2.COLOR_BGR2GRAY)
                #img = pp.preprocess(img)
                #showImage('img', img)
                #save the image
                X.append(img)
                Y.append(int(name[0]))
    
    return X, Y
                
#generateDataset(RAW) 

def standardize(X, Y):
    '''
    Here we flatten the 2D images in X and divide by 255.
    Y will be converted to one hot vector format.
    Standardizing dataset sppeds up the training process.
    '''
    X = np.array(X)
    Y = np.array(Y)
    X = X.reshape(X.shape[0], -1).T #each training example is flatten to 1D array
    X = X/255      #Each value will be between 0 and 1
    
    #convert Y to one hot vector
    num_labels = 62 #for 62 characters
    Y = encode_labels(Y, num_labels)
    
    return X, Y

               


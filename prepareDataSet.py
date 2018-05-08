#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 19:15:30 2018

@author: prashant
"""

'''
Create dataset for Odia Character Recognition.
'''
import os
import cv2
import sys
import math
import numpy as np

TrainFolder = '/home/prashant/Downloads/OdiaCharacterRecognition/Data/oriya_numeral'

#We want to image of size 16 x 16
HEIGHT = 32
WIDTH = 32

def showImage(name, img):
    '''
    Show image in a window.
    img: jpg image
    name: name of window in which img will be shown
    '''
    cv2.imshow(name, img)
    k = cv2.waitKey(0)
    #press q to exit
    if k == ord('q'):
        cv2.destroyWindow(name) #destroy only window with name 'name'
        sys.exit()
    cv2.destroyAllWindows()
    

def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty


def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted


#saveCharImage(TrainFolder)
        
def printFile(folder):
    for file in os.listdir(folder):
        print(file)
        path = os.path.join(folder,file)
        if os.path.isdir(path) == True:
            printFile(path)

#printFile(TrainFolder)
def resizeImage(im, w = None, h = None, scale=0.5):
    '''
    path: location of image.
    scale: by how much we want to scale the image.
    Keep the ASPECT RATIO while resizing.
    '''
    #read image
    (r, c) = im.shape[:2]
        
    
    #resize according to scale
    if w == None and h==None :
        (r, c) = im.shape[:2]
        r = int(round(r*scale))
        c = int(round(c*scale))
    else:
        if r > c:
            ratio = c/r
            r = h
            c = int(round(ratio * h))
        else:
            ratio = r/c
            c = w
            r = int(round(ratio * w))
    
        
    resized = cv2.resize(im, (c, r), cv2.INTER_AREA)
    return resized




'''
It's important that all images are of same dimension so that it can be
feed to the Neural Network .
'''
def preprocess(img):
    '''
    img: color image of single letter
    return: standardized image of single letter
    '''
    #binary threshold
    im = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    (thresh, gray) = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                
                
    #make image dimension as approx 20 x 20 so that
    # it can be made of size 28 x 28
    
    img = resizeImage(gray, w=20, h=20)
    print(img.shape)
    #blur = cv2.GaussianBlur(resized, (3,3), 0)
    
    #kernel = np.ones((2,2),np.uint8)
    #dilation = cv2.dilate(img,kernel,iterations = 1)
    #erosion = cv2.erode(dilation,kernel,iterations = 1)
    
    #pad with white pixel to make it's dimension 28x28
    rows, cols = img.shape
    colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
    rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
    img = np.lib.pad(img,(rowsPadding,colsPadding),'constant')
    print(img.shape)

    
    #center alphabets inside 28 x 28
    
    return img

   
def generateDataset(folder):
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
                img = preprocess(path)
                showImage('img', img)
                #save the image
                X.append(img)
                
        

#generateDataset(TrainFolder)
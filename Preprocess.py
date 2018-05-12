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
from scipy import ndimage

TrainFolder = '/home/prashant/Downloads/OdiaCharacterRecognition/Code/Data/oriya_numeral'

#We want to image of size 16 x 16
HEIGHT = 28
WIDTH  = 28

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
  

def saveImage(img, folder, cat, k):
    '''
    Save the given 'image' to the given folder with name as '1.jpg'
    img: processed image
    folder: where image will be stored
    k: Denotes the k'th example of given class
    
    '''
    
    #skip saving if n is pressed
    inp = input('Enter n if you do not want to save image or enter image class:')
    if inp != 'n':
        cv2.imwrite(folder + inp + '.'+str(k)+'.jpg', img)
    
    
    '''
    cv2.imwrite(folder + str(cat) + '.'+str(k)+'.jpg', img)
    '''
    

def getBestShift(img):
    '''
    This function returns the value (shiftx, shifty) .
    If each pixel is shifted by this amount then it will be
    centered in 28x28 box.
    '''
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx, shifty


def shift(img,sx,sy):
    '''
    Shift the location of each pixel by (sx, sy)
    '''
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted


#saveCharImage(TrainFolder)
        
def printFile(folder):
    '''
    Recursively print all file and folder of a given folder
    '''
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


def thinning(img):
    '''
    This function used in thinning of character.
    img: A binary image where background is of black color and character
    is of white color.
    Thinning increases the gap between the adjacent characters and
    helps in detecting the char contour easily.
    It also helps in standardizing the thick and thin character to thin
    characters.
    RETURN: skel, thinned or skelonized  image
    '''
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    done = False
    imgInvCp = img[:]
    while(not done):
        eroded = cv2.erode(imgInvCp, element) 
        dilate = cv2.dilate(eroded, element)
        subtracted = cv2.subtract(imgInvCp, dilate)
        skel = cv2.bitwise_or(skel, subtracted)
        imgInvCp = eroded[:]
    
        #Stop when every pixel is eroded
        if cv2.countNonZero(imgInvCp) == 0:
            done = True
    
    #close gaps in letters
    #closing = cv2.morphologyEx(skel, cv2.MORPH_CLOSE, element)
    
    #slight fatten the letter
    dilate = cv2.dilate(skel, element)
    return dilate


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
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #(thresh, gray) = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                
                
    #make image dimension as approx 20 x 20 so that
    # it can be made of size 28 x 28
    
    img = resizeImage(img, w=20, h=20)

    #blur = cv2.GaussianBlur(resized, (3,3), 0)
    
    #kernel = np.ones((2,2),np.uint8)
    #dilation = cv2.dilate(img,kernel,iterations = 1)
    #erosion = cv2.erode(dilation,kernel,iterations = 1)
    
    #pad with white pixel to make it's dimension 28x28
    rows, cols = img.shape
    colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
    rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
    img = np.lib.pad(img,(rowsPadding,colsPadding),'constant')
    
    #center the letter in 28 x 28 box
    sx, sy = getBestShift(img)
    img = shift(img, sx, sy)
    
    #thinning
    img = thinning(img)
    return img


                
        

#generateDataset(TrainFolder)
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
        sys.exit()
    cv2.destroyWindow(name) #destroy only window with name 'name'
    
    
def saveCharImage(folder):
    dataset = []
    for file in os.listdir(folder):
        path = os.path.join(folder,file)
        if os.path.isdir(path) == True:
            saveCharImage(path)
        else:
            #read only jpg file
            name = file.split('.')
            if name[-1] == 'jpg':
                #read image file
                im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                (thresh, gray) = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                
                '''
                #save to jpg
                name = file.split('.')
                print(folder)
                cv2.imwrite(folder+'/'+name[0]+'.jpg', im)
                '''
                #show image
                showImage('letter', gray)
                
                #resize
                (h, w) = gray.shape
                factor = w/h
                h = 20
                w = int(factor*20)
                print(h, w)
                #sys.exit()
                im = cv2.resize(gray, (w, h))
                
                #show resized image
                showImage('resized', im)
            
                #save the image
                dataset.append(im)
            
        

saveCharImage(TrainFolder)
        
def printFile(folder):
    for file in os.listdir(folder):
        print(file)
        path = os.path.join(folder,file)
        if os.path.isdir(path) == True:
            printFile(path)

#printFile(TrainFolder)
            


    
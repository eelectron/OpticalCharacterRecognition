#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 09:52:23 2018

@author: prashant
"""

'''
This program is used to create training dataset for
printed Odia alphabets and Numerals.
It extract single letter from image and save on disk for
further processng.
'''
import Preprocess as pp
import os
import cv2
import sys
import math
import numpy as np

TESTING2 = '/home/prashant/Downloads/OdiaCharacterRecognition/Code/Data/testing2/'
TESTING3 = '/home/prashant/Downloads/OdiaCharacterRecognition/Code/Data/testing3/'
TESTING4 = '/home/prashant/Downloads/OdiaCharacterRecognition/Code/Data/testing4/'
def sideSum(box):
    '''
    Sort the bounding box from left to right and top to bottom.
    Multiply y by 1000 so that box in y+1 level is farther than box at
    y level.
    '''
    
    return box[0] + 10*(box[1] + box[3])
    
def extractChar(img):
    #find contours of all chars
    '''
    Char must have WHITE color on black background then only
    findContours() function WILL WORK !!!
    '''
    
    #convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #binarize image
    _, bw = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    
    #erode for testing
    '''
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    dilate = cv2.dilate(bw, element)
    eroded = cv2.erode(dilate, element)
    bw = eroded[:]
    '''
    
    #finnd boundary of char
    copyBW = bw[:] #make copy
    im2, contours, hierarchy = cv2.findContours(copyBW, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #sort the contours so that box appears from top left to right bottom
    boundingBox = []
    for cnt in contours:
        boundingBox.append(cv2.boundingRect(cnt))
    boundingBox = sorted(boundingBox, key=sideSum)

    
    #draw bounding box
    bgr = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    
    #draw box around each char
    i = 0
    flag = False
    for box in boundingBox:
        x,y,w,h = box
        
        #get character from the image
        ch = bw[y:y+h, x:x+w]
        
        procImg =  pp.preprocess(ch)
        cv2.rectangle(bgr, (x,y), (x+w,y+h), (0,255,0), 1)
        pp.showImage('binairze', bgr)
        if flag == False and i == 11:
            flag = True
            continue
        pp.saveImage(procImg, TESTING4, i, 0)
        i += 1
        
        
#path = '/home/prashant/Downloads/OdiaCharacterRecognition/Code/Data/testing1/an1.png'
#img = cv2.imread(path)
#extractChar(img)
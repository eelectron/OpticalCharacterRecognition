#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 09:52:23 2018

@author: prashant
"""

'''
This program is used to create training dataset for
printed Odia alphabets and Numerals.
'''
import preprocess as pt
import os
import cv2
import sys
import math
import numpy as np

def sideSum(box):
    '''
    Sort the bounding box from left to right and top to bottom.
    Multiply y by 1000 so that box in y+1 level is farther than box at
    y level.
    '''
    return box[0] + 10*box[1]
    
def extractChar(img):
    #find contours of all chars
    '''
    Char must have WHITE color on black background then only
    findContours() function WILL WORK !!!
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    im2, contours, hierarchy = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #sort the contours so that box appears from top left to right bottom
    boundingBox = []
    for cnt in contours:
        boundingBox.append(cv2.boundingRect(cnt))
    boundingBox = sorted(boundingBox, key=sideSum)    
    
    #draw box around each char
    i=0
    for box in boundingBox:
        x,y,w,h = box
        ch = img[y:y+h, x:x+w]
        procImg =  pt.preprocess(ch)
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 1)
        pt.showImage('img', img)
        i = pt.saveImage(procImg, pt.DATASET)
        
        
path = '/home/prashant/Downloads/OdiaCharacterRecognition/Code/Data/images/alphaNumeric1.png'
img = cv2.imread(path)
extractChar(img)
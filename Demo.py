#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 12:24:30 2018

@author: prashant
"""

'''

'''
import prepareDatasetOfPrintedChar as pds
import generateDataset as gds
import NeuralNetForOdiaCharHelper as nnh
import numpy as np
import os
import cv2
import Preprocess as pp

characters = {0:2918, 1:2919, 2:2920 , 3:2921, 4:2922, 5:2923,
              6:2924, 7:2925, 8:2926, 9:2927, 10:2821, 11:2878,
              12:2823, 13:2824, 14:2825, 15:2826, 16:2827, 17:2912,
              18:2831, 19:2832, 20:2835, 21:2836, 22:2837, 23:2838, 24:2839, 
              25:2840,
              26:2841, 27:2842, 28:2843, 29:2844, 30:2845,  31:2846,
              32:2847, 33:2848, 34:2849,
              35:2850, 36:2851, 37:2852, 38:2853, 39:2854, 40:2855,
              41:2856, 42:2858, 43:2859,
              44:2860, 45:2861, 46:2862, 47:2863, 48:2864,
              49:2867, 50:2870, 51:2871, 52:2872, 
              53:2873, 54:0, 55:2911, 56:2866, 57:2908, 58:2909, 59:2818,
              61:2817, 60:2819}

'''
for key in characters.keys():
    print(chr(characters[key]), end=' ')
'''
#Real Test

#read each image from folder
TESTING3 =  '/home/prashant/Downloads/OdiaCharacterRecognition/Code/Data/testing3/'
TESTING4 =  '/home/prashant/Downloads/OdiaCharacterRecognition/Code/Data/testing4/'
for file in os.listdir(TESTING3):
    path = os.path.join(TESTING3,file)
    img = cv2.imread(path)
    pds.extractChar(img)
    
    
#Now generate dataset X, Y
X, Y = gds.generateDataset(TESTING4)
X, Y = gds.standardize(X, Y)


#Read parameters from pickle
pickle_in = open("para.pickle", "rb")
parameters = pickle.load(pickle_in)

print("Testing set accuracy: ",nnh.predict(X, Y, parameters))

AL, caches = L_model_forward(X, parameters)

for i in AL.shape[1]:
    cls = np.argmax(AL[:,i])
    print(chr(characters[cls]))
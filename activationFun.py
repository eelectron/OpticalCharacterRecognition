#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 20:29:50 2018

@author: prashant
"""
import numpy as np

def sigmoid(Z):
    return 1/(1+np.exp(-Z)), Z

def relu(Z):
    return np.maximum(Z, 0), Z


def sigmoidGradient(Z):
    a = 1/(1+np.exp(-Z))
    return np.multiply(a, (1 - a))


def reluGradient(Z):
    Z[Z>=0] = 1
    Z[Z<0] = 0
    return Z



def sigmoid_backward(dA, activation_cache):
    Z = activation_cache
    return np.multiply(dA, sigmoidGradient(Z))

def relu_backward(dA, activation_cache):
    Z = activation_cache
    return np.multiply(dA, reluGradient(Z))
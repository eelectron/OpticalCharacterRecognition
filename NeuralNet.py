#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 23:42:26 2018

@author: prashant
"""
import NeuralNetForOdiaCharHelper as nnh
import generateDataset as gd
import matplotlib.pyplot as plt
import numpy as np
import pickle
DATASET = '/home/prashant/Downloads/OdiaCharacterRecognition/Code/Data/training1/'
### CONSTANTS ###
layers_dims = [784, 300, 62]


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False): #lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization.
    ### START CODE HERE ###
    parameters = nnh.initialize_parameters_deep(layers_dims)
    ### END CODE HERE ###
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        ### START CODE HERE ### (≈ 1 line of code)
        AL, caches = nnh.L_model_forward(X, parameters)
        ### END CODE HERE ###
        
        # Compute cost.
        ### START CODE HERE ### (≈ 1 line of code)
        cost = nnh.compute_cost(AL, Y)
        ### END CODE HERE ###
    
        # Backward propagation.
        ### START CODE HERE ### (≈ 1 line of code)
        grads = nnh.L_model_backward(AL, Y, caches)
        ### END CODE HERE ###
 
        # Update parameters.
        ### START CODE HERE ### (≈ 1 line of code)
        parameters = nnh.update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters




X_orig, Y_orig = gd.generateDataset(DATASET)
train_x, train_y = gd.standardize(X_orig, Y_orig)
'''
#THESE OUR TRAINED PARAMETERS or WEIGHTS
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations=2500, print_cost=True)

#save parameters for later use
pickle_para = open("para.pickle", "wb")
pickle.dump(parameters, pickle_para)
pickle_para.close()
'''
#Read parameters from pickle
pickle_in = open("para.pickle", "rb")
parameters = pickle.load(pickle_in)

print("Training set accuracy: ",nnh.predict(train_x, train_y, parameters))



#Testing
TESTING2 = '/home/prashant/Downloads/OdiaCharacterRecognition/Code/Data/testing2/'
X_test_orig, Y_test_orig = gd.generateDataset(TESTING2)
test_x, test_y = gd.standardize(X_test_orig, Y_test_orig)
print("Testing set accuracy: ",nnh.predict(test_x, test_y, parameters))




#print corresponding character
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
AL, caches = nnh.L_model_forward(test_x, parameters)
for i in range(AL.shape[1]):
    cls = np.argmax(AL[:,i])
    act = np.argmax(test_y[:,i])
    print('Actual: ',chr(characters[act]), 'Predicted: ',chr(characters[cls]))
    
    

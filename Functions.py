import numpy as np
import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import value_and_grad 

#### Loss function
def Loss_function(point,Y):
    alpha = 5 # Define Alpha
    result = 0 # Define result for the Loss_function

    for m in range(len(Y)): # Run over all "edges"
        connection = Y[m][0] # Check to see if there is a connection
        point_1 = point[Y[m][1]] # Define coordinates for vertex 1
        point_2 = point[Y[m][2]] # Define coordinates for vertex 2
        distance = np.linalg.norm(point_1 - point_2) ** 2 # Calculate the euclidean distance squared
        sigmoid_value = 1 / (1 + np.exp(-connection * (alpha - distance))) # Calculate the sigmoid function
        result += np.log(sigmoid_value) # Sum all the probabilities 

    return result


#### Gradient function
def Gradient_function(point_number,index,Y,point):
    alpha = 5 # Define Alpha
    gradient = 0 # Define gradient for the Gradient_function
    for m in range(len(Y)): # Run over all "edges"
        if ((Y[m][1] == point_number) or (Y[m][2] == point_number)): # Check to see if point_number is part of the edge
            connection = Y[m][0] # Check to see if there is a connection
            distance = (np.linalg.norm(point[Y[m][1]] - point[Y[m][2]]))**2 # Calculate the euclidean distance squared
            point_index_diff = (point[point_number][index] - point[Y[m][1]][index]) + (point[point_number][index] - point[Y[m][2]][index]) # Calculate point_a,i-point_b,i
            numerator = -2 * connection * point_index_diff * np.exp(-connection * (alpha - distance)) # Calculate the numerator
            denominator = 1 + np.exp(-connection * (alpha - distance)) # Calculate the denominator
            gradient += numerator / denominator # Calculate the fraction and plus all connection there is with point_number together

    return gradient 
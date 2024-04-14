import numpy as np
import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import value_and_grad 
import math

############################################################ With Prior ############################################################
# Loss function with prior
## The Loss_function_prior take as input all the points, Y-matrix, dimension k and returns the MLL plus a prior
## For all possible edges (every row in Y) it takes the sigmod function to -y_{a,b} multipled with alhpha minus the distance between point_a and point_b
## Log of sigmoid_value is added to the constant: result, in every loop through the rows of Y
## Finally it add the prior, as the sum of alle the points multiplied with a constant
def Loss_function_prior(point,Y,k):
    alpha = 5 # Change alpha here
    result = 0

    for m in range(len(Y)):
        connection = Y[m][0] 
        point_a = point[Y[m][1]]
        point_b = point[Y[m][2]]
        distance = np.linalg.norm(point_a - point_b) ** 2 
        sigmoid_value = 1 / (1 + np.exp(-connection * (alpha - distance))) 
        result += np.log(sigmoid_value)
    prior = -1/2 * np.sum(np.square(list(point.values()))) * np.log(1/(2*math.pi)**(k/2))
    result += prior

    return result


# Gradient function with prior
## The Gradient_function_prior take as input the point number, the index, Y-matrix, all the points, dimension k and reuturns the gradient for the point number's index
## It loops over all possible edges (every row in Y), and checks if point_number is in that row, if so it calculates the gradient
## Firstly it calculates the numerator, where point_index_diff is the point number's index minus the other vertex point number's index
## The numerator is mulitplied with the sigmod function to -y_{a,b} multipled with alhpha minus the distance between point_a and point_b
## The fraction for the m row in Y is added to the constant gradient
## After the loop it add the gradient as minus the point number's index value multiplied with a constant
def Gradient_function_prior(point_number,index,Y,point,k):
    alpha = 5 # Change alpha here
    gradient = 0 
    for m in range(len(Y)):
        if ((Y[m][1] == point_number) or (Y[m][2] == point_number)):
            connection = Y[m][0]
            distance = (np.linalg.norm(point[Y[m][1]] - point[Y[m][2]]))**2
            point_index_diff = (point[point_number][index] - point[Y[m][1]][index]) + (point[point_number][index] - point[Y[m][2]][index])
            numerator = -2 * connection * point_index_diff * np.exp(-connection * (alpha - distance))
            denominator = 1 + np.exp(-connection * (alpha - distance))
            gradient += numerator / denominator
    
    gradient -=  np.abs(point[point_number][index])*np.log(1/(2*math.pi)**(k/2))
    return gradient 




########################################################### Without Prior ###########################################################
# Loss function without prior
## The Loss_function take as input all the points, Y-matrix and returns the MLL
## For all possible edges (every row in Y) it takes the sigmod function to -y_{a,b} multipled with alhpha minus the distance between point_a and point_b
## Log of sigmoid_value is added to the constant: result, in every loop through the rows of Y
def Loss_function(point,Y):
    alpha = 5 # Change alpha here
    result = 0

    for m in range(len(Y)):
        connection = Y[m][0] 
        point_a = point[Y[m][1]] 
        point_b = point[Y[m][2]] 
        distance = np.linalg.norm(point_a - point_b) ** 2 
        sigmoid_value = 1 / (1 + np.exp(-connection * (alpha - distance))) 
        result += np.log(sigmoid_value) 

    return result


# Gradient function
## The Gradient_function take as input the point number, the index, Y-matrix, all the points and reuturns the gradient for the point number's index
## It loops over all possible edges (every row in Y), and checks if point_number is in that row, if so it calculates the gradient
## Firstly it calculates the numerator, where point_index_diff is the point number's index minus the other vertex point number's index
## The numerator is mulitplied with the sigmod function to -y_{a,b} multipled with alhpha minus the distance between point_a and point_b
## The fraction for the m row in Y is added to the constant gradient
def Gradient_function(point_number,index,Y,point):
    alpha = 5 # Change alpha here
    gradient = 0 
    for m in range(len(Y)): 
        if ((Y[m][1] == point_number) or (Y[m][2] == point_number)):
            connection = Y[m][0] 
            distance = (np.linalg.norm(point[Y[m][1]] - point[Y[m][2]]))**2 
            point_index_diff = (point[point_number][index] - point[Y[m][1]][index]) + (point[point_number][index] - point[Y[m][2]][index])
            numerator = -2 * connection * point_index_diff * np.exp(-connection * (alpha - distance)) 
            denominator = 1 + np.exp(-connection * (alpha - distance)) 
            gradient += numerator / denominator 

    return gradient 



########################################################### Learning rate ###########################################################
# The first step is 0.1, and slowly decrease to near 0
def LR(x):
    return np.exp(-x/100)*0.1
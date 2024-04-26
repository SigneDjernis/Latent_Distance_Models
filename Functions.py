import numpy as np
import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import value_and_grad 
import math

############################################################ With Prior ############################################################
# Loss function with prior
## The Loss_function_prior take as input all the points, Y-matrix, dimension k, the distance metric, alpha and returns the MLL plus a prior
## For all possible edges (every row in Y) it takes the sigmod function to -y_{a,b} multipled with alhpha minus the distance between point_a and point_b
## Log of sigmoid_value is added to the constant: result, in every loop through the rows of Y
## Finally it add the prior, as the sum of alle the points plus a constant that depends on the dimension
def Loss_function_prior(point,Y,k,dis,alpha):
    result = 0

    for m in range(len(Y)):
        connection = Y[m][0] 
        point_a = point[Y[m][1]]
        point_b = point[Y[m][2]]
        if dis == "cosi":
            distance = np.dot(point_a,point_b)/(np.linalg.norm(point_a)*np.linalg.norm(point_b))
        if dis == "norm":
            distance = np.linalg.norm(point_a - point_b) ** 2 
        sigmoid_value = 1 / (1 + np.exp(-connection * (alpha - distance))) 
        result += np.log(sigmoid_value)
    prior = np.log(1/(2*math.pi)**(k/2)) - 1/2 * np.sum(np.square(list(point.values()))**2)
    result += prior

    return result


def Loss_function_prior_fast(point,Y,k,alpha):
    Y = np.array(Y)
    connections = Y[:, 0]
    indices_a = Y[:, 1]
    indices_b = Y[:, 2]

    points_a = [point[index] for index in indices_a]
    points_b = [point[index] for index in indices_b]
    points_a_np = np.array(points_a)
    points_b_np = np.array(points_b)

    distances = np.sum((points_a_np - points_b_np) ** 2, axis=1)

    sigmoid_value = 1 / (1 + np.exp(-connections * (alpha - distances)))
    prior = np.log(1/(2*math.pi)**(k/2)) - 1/2 * np.sum(np.square(list(point.values()))**2)
    return np.sum(np.log(sigmoid_value)) + prior




# Gradient function with prior
## The Gradient_function_prior take as input the point number, the index, Y-matrix, all the points, the distance metric, alpha and reuturns the gradient for the point number's index
## It loops over all possible edges (every row in Y), and checks if point_number is in that row, if so it calculates the gradient
## Firstly it calculates the numerator, where point_index_diff is the point number's index minus the other vertex point number's index
## The numerator is mulitplied with the sigmod function to -y_{a,b} multipled with alhpha minus the distance between point_a and point_b
## The fraction for the m row in Y is added to the constant gradient
## After the loop it add the gradient as minus the point number's index value
def Gradient_function_prior(point_number,index,Y,point,dis,alpha):
    gradient = 0 
    for m in range(len(Y)):
        if ((Y[m][1] == point_number) or (Y[m][2] == point_number)):
            connection = Y[m][0]
            point_a = point[Y[m][1]]
            point_b = point[Y[m][2]]
            if dis == "cosi":
                distance = np.dot(point_a,point_b)/(np.linalg.norm(point_a)*np.linalg.norm(point_b))
            if dis == "norm":
                distance = np.linalg.norm(point_a - point_b) ** 2
            point_index_diff = (point[point_number][index] - point_a[index]) + (point[point_number][index] - point_b[index])
            numerator = -2 * connection * point_index_diff * np.exp(-connection * (alpha - distance))
            denominator = 1 + np.exp(-connection * (alpha - distance))
            gradient += numerator / denominator
    
    gradient -=  point[point_number][index]
    return gradient 


def Gradient_function_prior_fast(point_number,index,Y,point,alpha):
    Y = np.array(Y)
    mask_a = (Y[:, 1] == point_number)
    mask_b = (Y[:, 2] == point_number)

    Y_edges = Y[mask_a+mask_b]
    connections = Y_edges[:, 0]
    indices_a = Y_edges[:, 1]
    indices_b = Y_edges[:, 2]

    points_a = [point[index] for index in indices_a]
    points_b = [point[index] for index in indices_b]
    points_a_np = np.array(points_a)
    points_b_np = np.array(points_b)

    distances = np.sum((points_a_np - points_b_np) ** 2, axis=1)

    mask = mask_a + mask_b
    point_index_diff = (points_a_np[:,index] - points_b_np[:,index]) * mask_a[mask] + (points_b_np[:,index] - points_a_np[:,index]) * mask_b[mask]

    numerators = -2 * connections * point_index_diff * np.exp(-connections * (alpha - distances)) 
    denominators = 1 + np.exp(-connections * (alpha - distances)) 

    return np.sum(numerators / denominators) - point[point_number][index]


########################################################### Without Prior ###########################################################
# Loss function without prior
## The Loss_function take as input all the points, Y-matrix, the distance metric, alpha and returns the MLL
## For all possible edges (every row in Y) it takes the sigmod function to -y_{a,b} multipled with alhpha minus the distance between point_a and point_b
## Log of sigmoid_value is added to the constant: result, in every loop through the rows of Y
def Loss_function(point,Y,dis,alpha):
    result = 0

    for m in range(len(Y)):
        connection = Y[m][0] 
        point_a = point[Y[m][1]] 
        point_b = point[Y[m][2]] 
        if dis == "cosi":
            distance = np.dot(point_a,point_b)/(np.linalg.norm(point_a)*np.linalg.norm(point_b))
        if dis == "norm":
            distance = np.linalg.norm(point_a - point_b) ** 2 
        sigmoid_value = 1 / (1 + np.exp(-connection * (alpha - distance))) 
        result += np.log(sigmoid_value) 

    return result


def Loss_function_fast(point,Y,alpha):
    Y = np.array(Y)
    connections = Y[:, 0]
    indices_a = Y[:, 1]
    indices_b = Y[:, 2]

    points_a = [point[index] for index in indices_a]
    points_b = [point[index] for index in indices_b]
    points_a_np = np.array(points_a)
    points_b_np = np.array(points_b)

    distances = np.sum((points_a_np - points_b_np) ** 2, axis=1)

    sigmoid_value = 1 / (1 + np.exp(-connections * (alpha - distances)))
    return np.sum(np.log(sigmoid_value))


# Gradient function
## The Gradient_function take as input the point number, the index, Y-matrix, all the points, the distance metric, alpha and reuturns the gradient for the point number's index
## It loops over all possible edges (every row in Y), and checks if point_number is in that row, if so it calculates the gradient
## Firstly it calculates the numerator, where point_index_diff is the point number's index minus the other vertex point number's index
## The numerator is mulitplied with the sigmod function to -y_{a,b} multipled with alhpha minus the distance between point_a and point_b
## The fraction for the m row in Y is added to the constant gradient
def Gradient_function(point_number,index,Y,point,dis,alpha):
    gradient = 0 
    for m in range(len(Y)): 
        if ((Y[m][1] == point_number) or (Y[m][2] == point_number)):
            connection = Y[m][0]
            point_a = point[Y[m][1]]
            point_b = point[Y[m][2]]
            if dis == "cosi":
                distance = np.dot(point_a,point_b)/(np.linalg.norm(point_a)*np.linalg.norm(point_b))
            if dis == "norm":
                distance = np.linalg.norm(point_a - point_b) ** 2
            point_index_diff = (point[point_number][index] - point_a[index]) + (point[point_number][index] - point_b[index])
            numerator = -2 * connection * point_index_diff * np.exp(-connection * (alpha - distance)) 
            denominator = 1 + np.exp(-connection * (alpha - distance)) 
            gradient += numerator / denominator 

    return gradient 


def Gradient_function_fast(point_number,index,Y,point,alpha):
    Y = np.array(Y)
    mask_a = (Y[:, 1] == point_number)
    mask_b = (Y[:, 2] == point_number)

    Y_edges = Y[mask_a+mask_b]
    connections = Y_edges[:, 0]
    indices_a = Y_edges[:, 1]
    indices_b = Y_edges[:, 2]

    points_a = [point[index] for index in indices_a]
    points_b = [point[index] for index in indices_b]
    points_a_np = np.array(points_a)
    points_b_np = np.array(points_b)

    distances = np.sum((points_a_np - points_b_np) ** 2, axis=1)

    mask = mask_a + mask_b
    point_index_diff = (points_a_np[:,index] - points_b_np[:,index]) * mask_a[mask] + (points_b_np[:,index] - points_a_np[:,index]) * mask_b[mask]

    numerators = -2 * connections * point_index_diff * np.exp(-connections * (alpha - distances)) 
    denominators = 1 + np.exp(-connections * (alpha - distances)) 

    return np.sum(numerators / denominators)



########################################################### Learning rate ###########################################################
# The first step is 0.1, and slowly decrease to near 0
def LR(x):
    return np.exp(-x/100)*0.1+0.0001
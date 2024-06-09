import numpy as np
import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import value_and_grad 
import math

############################################################ With Prior ############################################################
# Loss function with prior
# The Loss_function_prior_fast computes the loss (negative log-likelihood) with a prior.
# Inputs:
#   - point: A list of all points.
#   - Y: A matrix where each row represents a relation between two points.
#   - k: The dimension of the points.
#   - alpha: A parameter used in the sigmoid function.
# Returns:
#   - The computed loss value including a prior term.
def Loss_function_prior_fast(point,Y,k,alpha):
    Y = np.array(Y)
    connections = Y[:, 0]
    indices_a = Y[:, 1]
    indices_b = Y[:, 2]

    points_a = [point[i] for i in indices_a]
    points_b = [point[i] for i in indices_b]
    points_a_np = np.array(points_a)
    points_b_np = np.array(points_b)

    distances = np.sum((points_a_np - points_b_np) ** 2, axis=1)

    sigmoid_value = 1 / (1 + np.exp(-connections * (alpha - distances)))
    prior = np.log(1/(2*math.pi)**(k/2)) - 1/2 * np.sum(np.square(list(point.values())))
    return np.sum(np.log(sigmoid_value)) + prior


# Gradient function with prior
# The Gradient_function_prior_fast computes the gradient with a prior for a specific point in the data.
# Inputs:
#   - point_number: The point for which the gradient is being computed.
#   - dim: The dimension of the points.
#   - Y: A matrix where each row represents a relation between two points.
#   - point: A list of all points.
#   - alpha: A parameter used in the sigmoid function.
# Returns:
#   - The gradient with a prior vector for the given point_number.
def Gradient_function_prior_fast(point_number,dim,Y,point,alpha):
    Y = np.array(Y)
    mask_a = (Y[:, 1] == point_number)
    mask_b = (Y[:, 2] == point_number)

    Y_edges = Y[mask_a+mask_b]
    connections = Y_edges[:, 0]
    indices_a = Y_edges[:, 1]
    indices_b = Y_edges[:, 2]

    points_a = [point[i] for i in indices_a]
    points_b = [point[i] for i in indices_b]
    points_a_np = np.array(points_a)
    points_b_np = np.array(points_b)

    distances = np.sum((points_a_np - points_b_np) ** 2, axis=1)

    mask = mask_a + mask_b

    reshaped_mask_a = mask_a[mask][:, np.newaxis]
    reshaped_mask_b = mask_b[mask][:, np.newaxis]
    point_index_diff = (points_a_np[:,0:dim] - points_b_np[:,0:dim]) * reshaped_mask_a + (points_b_np[:,0:dim] - points_a_np[:,0:dim]) * reshaped_mask_b

    numerators = -2 * connections[:, np.newaxis] * point_index_diff * np.exp(-connections * (alpha - distances))[:, np.newaxis]
    denominators = 1 + np.exp(-connections * (alpha - distances)) 

    return (np.sum(numerators/denominators[:, np.newaxis], axis=0)) - point[point_number]


########################################################### Without Prior ###########################################################
# Loss function without prior
# The Loss_function_fast computes the loss (negative log-likelihood) without a prior.
# Inputs:
#   - point: A list of all points.
#   - Y: A matrix where each row represents a relation between two points.
#   - alpha: A parameter used in the sigmoid function.
# Returns:
#   - The computed loss value.
def Loss_function_fast(point,Y,alpha):
    Y = np.array(Y)
    connections = Y[:, 0]
    indices_a = Y[:, 1]
    indices_b = Y[:, 2]

    points_a = [point[i] for i in indices_a]
    points_b = [point[i] for i in indices_b]
    points_a_np = np.array(points_a)
    points_b_np = np.array(points_b)

    distances = np.sum((points_a_np - points_b_np) ** 2, axis=1)

    sigmoid_value = 1 / (1 + np.exp(-connections * (alpha - distances)))
    return np.sum(np.log(sigmoid_value))



# Gradient function without prior
# The Gradient_function_prior_fast computes the gradient without a prior for a specific point in the data.
# Inputs:
#   - point_number: The point for which the gradient is being computed.
#   - dim: The dimension of the points.
#   - Y: A matrix where each row represents a relation between two points.
#   - point: A list of all points.
#   - alpha: A parameter used in the sigmoid function.
# Returns:
#   - The gradient without a prior vector for the given point_number.
def Gradient_function_fast(point_number,dim,Y,point,alpha):
    Y = np.array(Y)
    mask_a = (Y[:, 1] == point_number)
    mask_b = (Y[:, 2] == point_number)

    Y_edges = Y[mask_a+mask_b]
    connections = Y_edges[:, 0]
    indices_a = Y_edges[:, 1]
    indices_b = Y_edges[:, 2]

    points_a = [point[i] for i in indices_a]
    points_b = [point[i] for i in indices_b]
    points_a_np = np.array(points_a)
    points_b_np = np.array(points_b)

    distances = np.sum((points_a_np - points_b_np) ** 2, axis=1)

    mask = mask_a + mask_b

    reshaped_mask_a = mask_a[mask][:, np.newaxis]
    reshaped_mask_b = mask_b[mask][:, np.newaxis]
    point_index_diff = (points_a_np[:,0:dim] - points_b_np[:,0:dim]) * reshaped_mask_a + (points_b_np[:,0:dim] - points_a_np[:,0:dim]) * reshaped_mask_b

    numerators = -2 * connections[:, np.newaxis] * point_index_diff * np.exp(-connections * (alpha - distances))[:, np.newaxis]
    denominators = 1 + np.exp(-connections * (alpha - distances)) 

    return (np.sum(numerators/denominators[:, np.newaxis], axis=0))


############################################################# Baseline #############################################################
# Baseline
# Computes the probability of connections between two vertices.
# Inputs:
#   - Y: The original matrix where each row represents a relation between two points.
#   - test_number: The index of the relation being tested.
#   - Y_updated: The updated training matrix of relations.
# Returns:
#   - An array containing probability for each relation between the vertices connected in Y[test_number].
def Baseline_function(Y,test_number,Y_updated):
    vertex_a = Y[test_number, 1]
    vertex_b = Y[test_number, 2]
    mask_conncted = Y_updated[:,0] == 1
    Y_connected = Y_updated[mask_conncted]

    prop = np.zeros(len(vertex_a))
    
    for k in range(len(vertex_a)):
        Y_a_1 = Y_connected[:,1] == vertex_a[k]
        Y_a_2 = Y_connected[:,2] == vertex_a[k]
        Y_a = Y_a_1 + Y_a_2
        degree_a = Y_connected[Y_a,1] + Y_connected[Y_a,2] - vertex_a[k]

        Y_b_1 = Y_connected[:,1] == vertex_b[k]
        Y_b_2 = Y_connected[:,2] == vertex_b[k]
        Y_b = Y_b_1 + Y_b_2
        degree_b = Y_connected[Y_b,1] + Y_connected[Y_b,2] - vertex_b[k]

        prop[k] = 2*len(np.intersect1d(degree_a, degree_b))/(len(degree_a) + len(degree_b))

    return prop





########################################################### Learning rate ###########################################################
# The first step is 0.1, and slowly decrease to near 0
def LR(x):
    return np.exp(-x/100)*0.1+0.0001
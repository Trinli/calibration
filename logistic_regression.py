"""
Logistic regression
-only focusing on the univariate case with binary class, a.k.a Platt-scaling (Platt, 1998).
-because scipy only provides regularized versions, and because I can.
"""

from random import gauss
import numpy as np


def sigmoid(x):
    return(1 / (1 + np.exp(-x)))


def train_logistic_regression(data_class,
                              data_scores,
                              learning_rate=5.0,
                              convergence_criterion=1e-6,
                              max_iterations=1e5):
    """
    Function for training a logistic regression model.
    
    Args:
    data_class (np.array([])): An array of class-labels for samples. True
     indicates a positive sample.
    data_scores (np.array([])): An array of floats for samples.
    leaning_rate (float): Learning rate for the algorithm
    convergence_criterion (float): If new gradients are below this value, the
     algorithm will stop
    max_iterations (int): Maximum number of iterations before stop.
    """
    # 1. Randomly assign values to b_0 and b_1
    b_0 = gauss(0, 1)
    b_1 = gauss(0, 1)
    i = 0
    log_likelihood = -np.inf
    while(True):
        # 2. Estimate gradients for b_0 and b_1 for current values and data
        # IS THERE SOME NUMERIC INSTABILITY HERE?
        b_0_d = 1 / len(data_class) * sum([item_class - sigmoid(b_0 + b_1 * item_score) for item_class, item_score in zip(data_class, data_scores)])
        b_1_d = 1 / len(data_class) * sum([(item_class - sigmoid(b_0 + b_1 * item_score)) * item_score for item_class, item_score in zip(data_class, data_scores)])
        # 3. Update b_0 and b_1 in the positive direction of the gradients (maximize!)
        b_0 += learning_rate * b_0_d
        b_1 += learning_rate * b_1_d
        # 4. Check for convergence or max iterations, if not, return to 2.
        log_likelihood_tmp = get_log_likelihood(data_class, data_scores, b_0, b_1)
        # print(log_likelihood_tmp, b_0, b_1, b_0_d, b_1_d)
        if (i == max_iterations):
            print("Max. iterations reached.")
            break
        elif np.abs(b_0_d) < convergence_criterion and np.abs(b_1_d) < convergence_criterion:
            break
        log_likelihood = log_likelihood_tmp
        i += 1
    # 5. Return model
    return({'b_0': b_0, 'b_1': b_1})


def predict_logistic_regression(model, data_scores):
    """
    Function for prediction with a logistic regression model
    
    Args:
    model (model as produced by train_logistic_regression): Model to use for predictions
    data_scores (np.array([])): Array of scores to transform into probabilities.
    """
    probabilities = np.array([sigmoid(model['b_0'] + model['b_1'] * item_score) for item_score in data_scores])
    return(probabilities)


def get_log_likelihood(data_class, data_scores, b_0, b_1):
    # sum_i(y_i*log(F(x_i)) + (1-y_i)*(log(1-F(x_i))))
    # Only produces nan's. Quite useless...
    log_likelihood = 1 / len(data_class) * sum([item_class * np.log(sigmoid(b_0 + b_1 * item_score)) +
                                                (1 - item_class) * np.log(1 - sigmoid(b_0 + b_1 * item_score))
                                                for item_class, item_score in zip(data_class, data_scores)])
    return(log_likelihood)

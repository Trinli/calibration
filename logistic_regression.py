"""
Logistic regression
-only focusing on the univariate case with binary class
-because scipy only provides regularized versions, and because I can.
"""

from random import gauss
import numpy as np


def sigmoid(x):
    return(1 / (1 + np.exp(-x)))

# # Results with learning rate 1, and convergence criteria 1e-6
# In [217]: isotonic.mean_squared_error(data_class, logistic_probabilities)
# Out[217]: 0.038820194788117643
# # Convergence criteria 1e-4
# In [222]: isotonic.mean_squared_error(data_class, logistic_probabilities)
# Out[222]: 0.038820050003216117
# Hence we seen that from MSE perspective, the convergence criterion is not super crucial.


def train_logistic_regression(data_class, data_scores, learning_rate=5.0, convergence_criteria=1e-6, max_iterations=1e5):
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
        elif np.abs(b_0_d) < convergence_criteria and np.abs(b_1_d) < convergence_criteria:
            break
        log_likelihood = log_likelihood_tmp
        i += 1
    # 5. Return model
    return({'b_0': b_0, 'b_1': b_1})


def predict_logistic_regression(model, data_scores):
    probabilities = np.array([sigmoid(model['b_0'] + model['b_1'] * item_score) for item_score in data_scores])
    return(probabilities)


def get_log_likelihood(data_class, data_scores, b_0, b_1):
    # sum_i(y_i*log(F(x_i)) + (1-y_i)*(log(1-F(x_i))))
    # Only produces nan's. Quite useless...
    log_likelihood = 1 / len(data_class) * sum([item_class * np.log(sigmoid(b_0 + b_1 * item_score)) +
                                                (1 - item_class) * np.log(1 - sigmoid(b_0 + b_1 * item_score))
                                                for item_class, item_score in zip(data_class, data_scores)])
    return(log_likelihood)

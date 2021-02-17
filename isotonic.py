"""
Some code to test reliably calibrated isotonic regression.
sklearn.isotonic.IsotonicRegression produces a function that
interpolates values falling _between_ bins of the calibration
set.
Although it sounds plausible, there is no scientific justification
for this (no academic paper on it, only for prediction). Initial
tests indicate that it produces better testing set auc-roc and
mse this way instead of splitting the gaps between neighboring bins.
However, from theoretical analysis, splitting the gaps between
neighboring bins is easier as all input scores and output probabilities
can then be modeled using Beta-distributions.
"""

import numpy as np
import pickle
from scipy.interpolate import interp1d
from sklearn.isotonic import IsotonicRegression
from scipy.stats import beta
from scipy.special import gamma, gammaln
from scipy.special import beta as beta_fn
from sklearn.metrics import roc_auc_score
import matplotlib
# Force matplotlib not to use any Xwindow backend:
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from sklearn.isotonic import IsotonicRegression


def load_pickle(file_name='./data.pickle'):
    # Function for loading files in pickle format.
    with open(file_name, 'rb') as handle:
        data = pickle.load(handle, encoding='latin1')
    return(data)


def save_pickle(data, file_name='./tmp/data.pickle'):
    with open(file_name, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def predict(model, data_scores):
    try:  # IsotonicRegression model
        data_probabilities = model.predict(T=data_scores)
    except:
        try:  # Interpolation model
            data_probabilities = model(data_scores)
        except:  # kNN-model
            data_probabilities = model.predict_proba(X=data_scores)[:, 1]
    return(data_probabilities)


def estimate_performance(model, data_class, data_scores):
    """
    Function for estimating performance metrics (AUC-ROC and MSE)
    of model.
    
    Args:
    model {IsotonicRegression, interpolation omdel, kNN-model}:
     model used to convert scores to probabilities.
    data_class (np.array([])): Array of class labels. True indicates
     positive sample, False negative.
    data_scores (np.array([])): Scores produced e.g. by machine
     learning model for samples.
    """
    data_probabilities = predict(model. data_scores)
    # Estimate mean squared error:
    mse = sum((data_class - data_probabilities)**2) / len(data_class)
    # Estimate AUC-ROC:
    auc_roc = roc_auc_score(data_class, data_probabilities)
    res = dict([('mse', mse), ('auc_roc', auc_roc)])
    return(res)


def credible_interval(k, n, confidence_level=.95, tolerance=1e-6):
    """
    Auxiliary function for estimating width of credible interval.
    Finds the highest posterior density interval using binary search.
    
    Args:
    k (int): Number of positive samples
    n (int): Number of samples
    confidence_level (float): Probability mass that has to fall within
     the credible intervals. In ]0, 1].
    tolerance (float): Upper limit for tolerance of probability mass
     within credible interval.
    """
    p_min_lower = float(0)
    p_middle = p_min_upper = p_max_lower = k / float(n)
    p_max = p_max_upper = float(1)
    p_min_middle = (p_min_lower + p_middle) / 2  # == 0 if k == 0.
    p_max_middle = (p_middle + p_max) / 2  # == n if k == n
    if(k == 0):  # Exception handling
        # p_min_middle = 0  # Per definition... it's the peak.
        while(abs(beta.cdf(p_max_middle, 1, n + 1) - confidence_level) > tolerance):
            if(beta.cdf(p_max_middle, 1, n + 1) > confidence_level):
                p_max_upper = p_max_middle
            else:
                p_max_lower = p_max_middle
            p_max_middle = (p_max_lower + p_max_upper) / 2
    elif(k == n):  # Exception handling
        while(abs(1 - beta.cdf(p_min_middle, k + 1, 1) - confidence_level) > tolerance):
            if(1 - beta.cdf(p_min_middle, k + 1, 1) > confidence_level):
                p_min_lower = p_min_middle
            else:
                p_min_upper = p_min_middle
            p_min_middle = (p_min_lower + p_min_upper) / 2
    else:  # Main case
        while(abs(beta.cdf(p_max_middle, k + 1, n - k + 1) - beta.cdf(p_min_middle, k + 1, n - k + 1) -
                  confidence_level) > tolerance / 2):
            # Binary search
            # Reset p-max values for new iteration:
            p_max_lower = p_middle
            p_max_upper = p_max
            p_max_middle = (p_max_lower + p_max_upper) / 2
            while(abs(beta.logpdf(p_min_middle, k + 1, n - k + 1) -
                      beta.logpdf(p_max_middle, k + 1, n - k + 1)) > tolerance / 2):
                # Binary search to find p_max corresponding to p_min (same value in pdf).
                if(k * np.log(p_min_middle) + (n - k) * np.log(1 - p_min_middle) >
                   k * np.log(p_max_middle) + (n - k) * np.log(1 - p_max_middle)):
                    p_max_upper = p_max_middle
                else:
                    p_max_lower = p_max_middle
                p_max_middle = (p_max_lower + p_max_upper) / 2
            if(beta.cdf(p_max_middle, k + 1, n - k + 1) - beta.cdf(p_min_middle, k + 1, n - k + 1) >
               confidence_level):
                p_min_lower = p_min_middle
            else:
                p_min_upper = p_min_middle
            p_min_middle = (p_min_lower + p_min_upper) / 2
    return(dict([('p_min', p_min_middle), ('p_max', p_max_middle)]))


def modify_model(model):
    """
    Auxiliary function for reliably calibrated isotonic regression.
    By default, sklearn.isotonic.IsotonicRegression produces an interpolation
    model that is piecewise constant, but where the values falling between
    bins are determined by an interpolation between two bins instead of being
    mapped to one of the bins.
    This function removes the gaps between the bins resulting in an 
    interpolation model that is piecewise constant for all x.
    By setting the next bin boundary equal to the previous one, we define the
    space entirely (leaving no gaps). The lower boundary belongs to *this bin,
    the upper to the next.
    
    Args:
    model (IsotonicRegression): Model as produced by sklearn.isotonic.IsotonicRegression
     or interpolation model.
    """
    try:
        # If model is of class sklearn.isotonic.IsotonicRegression
        x = model.f_.x
        y = model.f_.y
    except:
        # If model is of class scipy.interpolate.interp1d
        x = model.x
        y = model.y
    if(y[0] == y[1]):  # The borders are sometimes "off by one"...
        for i in range(int(len(x) / 2)):
            # Make corrections from y[1] & x[1]
            try:
                x_new = (x[i * 2 + 1] + x[i * 2 + 2]) / 2.0
                x[i * 2 + 1] = x_new
                x[i * 2 + 2] = x_new
            except IndexError:  # If there is an odd number of items in x
                pass
    else:  # y[0] != y[1]
        for i in range(int(len(x) / 2)):
            # Make corrections from y[0] & x[0]
            # This won't always work for x[-1]
            try:
                x_new = (x[i * 2] + x[i * 2 + 1]) / 2.0
                x[i * 2] = x_new
                x[i * 2 + 1] = x_new
            except IndexError:  # If there is an odd number of items in x
                pass
    interpolation_model = interp1d(x=x, y=y, bounds_error=False, assume_sorted=True)
    interpolation_model._fill_value_below = min(y)
    interpolation_model._fill_value_above = max(y)
    return(interpolation_model)


def plot_intervals(data, credible_intervals,
                   file_name="plot.png",
                   label="Credible intervals"):
    """
    Function for plotting the score-probability mapping with
    credible intervals.
    """
    # Plot points
    plt.axes().set_aspect('equal')
    plt.xlim([0, 1.01])
    plt.ylim([0, 1.01])
    plt.plot(data[0], data[0], 'kx')  # x's on the diagonal, in black.
    plt.title(label, fontsize=20)  # size="x-large")
    plt.xlabel("Bin probability", size="x-large")
    plt.ylabel("95% credible intervals", size="x-large")
    credible_lower = [row['p_min'] for row in credible_intervals]
    credible_upper = [row['p_max'] for row in credible_intervals]
    for i in range(len(data[0])):
        # Add black lines for confidence level
        plt.plot([data[0][i], data[0][i]], [credible_lower[i], credible_upper[i]], 'k-')
    # Add credible intervals
    plt.savefig(file_name)
    plt.gcf().clear()


def plot_reliability_diagram(model, data_scores, data_class,
                             label="Reliability diagram",
                             file_name="reliability_diagram.png"):
    # Function for plotting reliability diagrams.
    # Remove the gaps between bins:
    tmp_model = modify_model(model)
    # Testing set predicted probabilities vs. realized probabilities
    data_probabilities = tmp_model(data_scores)
    predicted_prob = np.unique(data_probabilities)
    true_prob = []
    for prob in predicted_prob:
        idx = (data_probabilities == prob)
        # Count number of positives
        tmp = sum(data_class[idx] == 1)
        # Estimate relative frequency:
        if(sum(idx) > 0):
            true_prob.append(tmp / float(sum(idx)))
        else:
            true_prob.append(0)
    plt.axes().set_aspect('equal')
    plt.xlim([0, 1.01])
    plt.ylim([0, 1.01])
    plt.title(label, fontsize=20)  # size="x-large")
    plt.xlabel("Bin probability", size="x-large")
    plt.ylabel("Actual frequency in testing set", size="x-large")
    plt.plot(predicted_prob, true_prob, 'kx')
    plt.plot([0, 1], [0, 1], 'b-')
    plt.savefig(file_name)
    plt.gcf().clf()


def correct_for_point_bins(x, y):
    """
    Auxiliary function for reliably calibrated isotonic regression
    (train_rcir).
    Iff there is e.g. 2 samples that map to 0.5 and they have different
    classes but the exact same scores, then these will create a bin
    with zero width and only one entry in x and y for the isotonic regression
    and the interpolation model. Further functions don't know how to handle
    this, so we will correct for that in this function.
    """
    if(y[0] == y[1]):
        for i in range(int(len(y) / 2)):
            try:
                if(y[2 * i] == y[2 * i + 1]):
                    pass  # Everything in order.
                else:
                    # We found a bin represented by only _one_ entry in x and y. Fix that.
                    y = np.insert(y, 2 * i + 1, y[2 * i])
                    x = np.insert(x, 2 * i + 1, x[2 * i])
            except IndexError:
                pass  # Last element. Everything in order.
    elif(y[0] != y[1]):
        for i in range(int(len(y) / 2)):
            try:
                if(y[2 * i + 1] == y[2 * i + 2]):
                    pass
                else:
                    y = np.insert(y, 2 * i + 2, y[2 * i + 1])
                    x = np.insert(x, 2 * i + 2, x[2 * i + 1])
            except IndexError:
                pass  # Everything in order.
    return({'x': x, 'y': y})


def expected_calibration_error(data_class, data_probabilities, k=10):
    """
    This function calculates the expected calibration error (ECE) as
    described in Naeini & al. 2015 with k=10 by default..

    Args:
    data_class (np.array([])): Array of class of data. True indicates positive
     class, False negative.
    data_probabilities (np.array([])): Array of predicted probabilities.
    k (int): Number of bins to use for estimating ECE. Needs to be in
     some sort of proportion to dataset size.
    """
    order_idx = np.argsort(data_probabilities)
    ordered_data_class = data_class[order_idx]
    ordered_data_probabilities = data_probabilities[order_idx]
    n_samples = data_class.shape[0]
    score = 0.0
    for i in range(k):
        # Rate of positives in slice
        o_i = np.mean(ordered_data_class[int(i * n_samples / k): int((i + 1) * n_samples / k)])
        # Average predicted probability
        e_i = np.mean(ordered_data_probabilities[int(i * n_samples / k): int((i + 1) * n_samples / k)])
        score += float(abs(o_i - e_i)) / k
    return(score)


def maximum_calibration_error(data_class, data_probabilities, k=10):
    """
    This function calculates the maximum calibration error (MCE) as described
    by Naeini & al. 2015 with k=10 by default.
    """
    order_idx = np.argsort(data_probabilities)
    ordered_data_class = data_class[order_idx]
    ordered_data_probabilities = data_probabilities[order_idx]
    n_samples = data_class.shape[0]
    max_err = 0.0
    for i in range(k):
        o_i = np.mean(ordered_data_class[int(i * n_samples / k): int((i + 1) * n_samples / k)])
        e_i = np.mean(ordered_data_probabilities[int(i * n_samples / k): int((i + 1) * n_samples / k)])
        err = abs(o_i - e_i)
        if(err > max_err):
            max_err = err
    return(max_err)


def accuracy(data_class, data_probabilities):
    return(np.mean(abs(data_class - data_probabilities)))


def mean_squared_error(data_class, data_probabilities):
    return(sum((data_class - data_probabilities) ** 2) / len(data_class))


def expected_bin_error(data_class, data_probabilities):
    # Ad hoc version that focuses on average error of prediction within bins.
    # Variant of expected calibration error.
    # Should the weighted averaging be done somehow?
    # Could we instead estimate likelihood of model given data?
    probabilities = np.unique(data_probabilities)
    err = 0.0
    for prob in probabilities:
        idx = data_probabilities == prob
        e_i = np.mean(data_class[idx])  # Empirical frequency
        # print("Mapping probability: " + str(prob) + ", empiric probability: " + str(e_i))
        # Weighted average of error:
        err += float(sum(data_class[idx])) / len(data_class) * e_i
    return(err)


def bootstrap_isotonic_regression(data_class, data_scores,
                                  sampling_rate=.95, n_models=200,
                                  y_min=0, y_max=1):
    # Function for training an ensemble of isotonic regression models
    models = []
    n_samples = data_class.shape[0]
    for i in range(n_models):
        tmp_model = IsotonicRegression(y_min=y_min, y_max=y_max, out_of_bounds='clip')
        idx = np.random.randint(low=0, high=n_samples, size=int(sampling_rate * n_samples))
        tmp_data_class = data_class[idx]
        tmp_data_scores = data_scores[idx]
        tmp_model.fit(X=tmp_data_scores, y=tmp_data_class)
        models.append(tmp_model)
    return(models)


def bootstrap_isotonic_regression_predict(models, data_scores):
    # Function for predicting from ensemble of models
    # Could this be averaged over model likelihood in a similar fashion as Naeini's model (BBQ)?
    probabilities = np.mean([model.predict(data_scores) for model in models], axis=0)
    return(probabilities)


def train_rcir(training_class, training_scores,
               credible_level=.95, d=1, y_min=0,
               y_max=1, merge_criterion='auc_roc'):
    """
    Function for training reliably calibrated isotonic regression (RCIR)
    as described in the upcoming PAKDD 2021-paper.

    Args:
    training_class (np.array([])): Array of classes for training data.
    training_scores (np.array([])): Array of scores for training data
    credible_level (float): Probability mass required to fall within
     credible interval. In [0, 1].
    d (float): Width of largest allowed credible interval containing
     probability mass specified by credible_level. In ]0, 1].
    y_min (float): Value of smallest prediction allowed by model. Here
     to allow preventing problems from arising due to zero-probabilities
     when estimating e.g. likelihoods. In [0,1[
    y_max (float): Largest prediction allowed by model. Value in ]y_min, 1].
    merge_criterion {'auc_roc', 'mse'}: Criterion to use for bin merges
     after isotonic regression has resolved all monotonicity conflicts.
    """
    isotonic_regression_model = IsotonicRegression(y_min=y_min, y_max=y_max, out_of_bounds='clip')
    isotonic_regression_model.fit(X=training_scores, y=training_class)
    # Extract the interpolation model we need:
    tmp_x = isotonic_regression_model.f_.x
    tmp_y = isotonic_regression_model.f_.y
    # Do some corrections (if there are any)
    tmp = correct_for_point_bins(tmp_x, tmp_y)
    x = tmp['x']
    y = tmp['y']
    # Use new boundaries to create an interpolation model that does
    # the heavy lifting of reliably calibrated isotonic regression:
    interpolation_model = interp1d(x=x, y=y, bounds_error=False)
    interpolation_model._fill_value_below = min(y)
    interpolation_model._fill_value_above = max(y)
    training_probabilities = interpolation_model(training_scores)
    # The following array contains all information defining the 
    # IR transformation
    bin_summary = np.unique(training_probabilities, return_counts=True)
    credible_intervals = [credible_interval(np.round(p * n), n) for (p, n) in
                          zip(bin_summary[0], bin_summary[1])]
    width_of_intervals = np.array([row['p_max'] - row['p_min'] \
        for row in credible_intervals])
    rcir_model = {'model': interpolation_model,
                  'credible level': credible_level,
                  'credible intervals': credible_intervals,
                  'width of intervals': width_of_intervals,
                  'bin summary': bin_summary,
                  'd': d}
    while(max(rcir_model['width of intervals']) > d):
        # Merge one more bin.
        rcir_model = merge_bin(rcir_model, training_class, training_scores,
                               merge_criterion)
    return(rcir_model)


def train_rcir_cv(training_class, training_scores,
                  validation_class, validation_scores,
                  credible_level=.95, y_min=0, y_max=1,
                  merge_criterion='auc_roc'):
    """
    Variant of reliably calibrated isotonic regression where the maximum
    allowed credible interval width is defined by maximum performance
    on validation set.
    
    Args:
    (see train_rcir())
    validation_class (np.array([])): Array of class labels for validation set.
    validation_score (np.array([])): Array of scores for validation set.
    """
    isotonic_regression_model = IsotonicRegression(y_min=y_min, y_max=y_max, out_of_bounds='clip')
    isotonic_regression_model.fit(X=training_scores, y=training_class)
    models = []
    # Extract the interpolation model we need:
    tmp_x = isotonic_regression_model.f_.x
    tmp_y = isotonic_regression_model.f_.y
    # Do some corrections (if there are any)
    tmp = correct_for_point_bins(tmp_x, tmp_y)
    x = tmp['x']
    y = tmp['y']
    # Use new boundaries to create an interpolation model that does the heavy lifting of
    # reliably calibrated isotonic regression:
    interpolation_model = interp1d(x=x, y=y, bounds_error=False)
    interpolation_model._fill_value_below = min(y)
    interpolation_model._fill_value_above = max(y)
    training_probabilities = interpolation_model(training_scores)
    # The following array contains all information defining the IR transformation
    bin_summary = np.unique(training_probabilities, return_counts=True)
    credible_intervals = [credible_interval(np.round(p * n), n) for (p, n) in
                          zip(bin_summary[0], bin_summary[1])]
    width_of_intervals = np.array([row['p_max'] - row['p_min'] for row in credible_intervals])
    rcir_model = {'model': interpolation_model, 'credible level': credible_level,
                  'credible intervals': credible_intervals, 'width of intervals': width_of_intervals,
                  'bin summary': bin_summary, 'd': -1}
    metrics = estimate_performance(rcir_model['model'], validation_class, validation_scores)
    models.append([0, rcir_model['model'], metrics])
    while(len(rcir_model['width of intervals']) > 2):  # There still exists bins to merge
        rcir_model = merge_bin(rcir_model, training_class, training_scores, merge_criterion)
        metrics = estimate_performance(rcir_model['model'], validation_class, validation_scores)
        models.append([0, rcir_model['model'], metrics])
    best_model_idx = [item[2]['auc_roc'] for item in models].index(max([item[2]['auc_roc'] for item in models]))
    return(models[best_model_idx][1])


def merge_bin(rcir_model, data_class, data_scores, merge_criterion='auc_roc'):
    """
    Auxiliary function for train_rcir. Performs one bin merge.
    Function could be hidden.
    """
    width_of_intervals = rcir_model['width of intervals']
    x = rcir_model['model'].x
    y = rcir_model['model'].y
    bin_summary = rcir_model['bin summary']
    credible_intervals = rcir_model['credible intervals']
    drop_idx = width_of_intervals.tolist().index(max(width_of_intervals))
    if drop_idx == 0:  # Exception handling. Fist bin has largest credible interval.
        # remove first two elements in x and y, update new first elements,
        # remove first element from width_of_intervals
        # and remove first elements from bin_summary[0] and bin_summary[1]
        y = np.delete(y, [0, 1])  # Drop first and second items.
        x = np.delete(x, [0, 1])
        new_prob = (bin_summary[0][0] * bin_summary[1][0] + bin_summary[0][1] * bin_summary[1][1]) / (bin_summary[1][0] + bin_summary[1][1])
        y[0] = new_prob
        try:  # y[1] doesn't exist if this is also the last bin.
            y[1] = new_prob
        except IndexError:
            pass
        # Leave x as is. bin_summary and width_of_intervals handled at end of loop.
        int_mod = interp1d(x, y, bounds_error=False)
        int_mod._fill_value_below = 0
        int_mod._fill_value_above = 1
        # print("Test-, and training performance, " + str(i) + " bins removed.d")
        # print(isotonic.estimate_performance(int_mod, test_class, test_scores))
        # print(isotonic.estimate_performance(int_mod, training_class, training_scores))
        tmp = credible_interval(k=round(bin_summary[0][0] * bin_summary[1][0] + bin_summary[0][1] * bin_summary[1][1]),
                                n=(bin_summary[1][0] + bin_summary[1][1]))
        width_of_intervals[0] = tmp['p_max'] - tmp['p_min']
        credible_intervals.pop(drop_idx)  # Remove line from credible intervals
        credible_intervals[0] = tmp
        bin_summary[0][1] = new_prob
        bin_summary[1][1] = bin_summary[1][0] + bin_summary[1][1]
    elif drop_idx == len(width_of_intervals) - 1:
        # More exception handling. '-1' for last element?
        # remove last element (only one!) of x and y:
        two_y_end = False
        if(y[-1] == y[-2]):
            two_y_end = True
        y = np.delete(y, drop_idx * 2)
        y = np.delete(y, drop_idx * 2 - 1)
        x = np.delete(x, drop_idx * 2)
        x = np.delete(x, drop_idx * 2 - 1)
        new_prob = (bin_summary[0][-1] * bin_summary[1][-1] + bin_summary[0][-2] * bin_summary[1][-2]) / (bin_summary[1][-1] + bin_summary[1][-2])
        # Hmm, there might be two bins for this y
        if(two_y_end):
            y[-2] = new_prob
        y[-1] = new_prob
        tmp = credible_interval(k=round(bin_summary[0][-1] * bin_summary[1][-1] + bin_summary[0][-2] * bin_summary[1][-2]),
                                n=(bin_summary[1][-1] + bin_summary[1][-2]))
        width_of_intervals[-2] = tmp['p_max'] - tmp['p_min']
        credible_intervals.pop(-1)  # Drop last.
        credible_intervals[-1] = tmp
        bin_summary[0][-2] = new_prob
        bin_summary[1][-2] = bin_summary[1][-1] + bin_summary[1][-2]
        # if((drop_idx != len(width_of_intervals)) or (drop_idx != 0):  # Main handling
        int_mod = interp1d(x, y, bounds_error=False)
        int_mod._fill_value_below = 0
        int_mod._fill_value_above = 1
        # print("Testing set performance, " + str(i) + " bins removed.c")
        # print(isotonic.estimate_performance(int_mod, test_class, test_scores))
    else:
        # Main method, i.e. when we are not dealing with the first or last bin.
        # y contains the probability to be dropped twice
        y = np.delete(y, drop_idx * 2 + 1)
        y = np.delete(y, drop_idx * 2)
        # Test lower:
        x_tmp_lower = np.array(x)  # Create NEW array!!
        x_tmp_lower = np.delete(x_tmp_lower, drop_idx * 2)  # Lower boundary of *this bin
        x_tmp_lower = np.delete(x_tmp_lower, drop_idx * 2 - 1)  # Upper boundary of smaller bin
        y_tmp_lower = np.array(y)  # Create _new_ array!!!
        new_prob_lower = ((bin_summary[1][drop_idx] * bin_summary[0][drop_idx] +
                          bin_summary[1][drop_idx - 1] * bin_summary[0][drop_idx - 1]) /
                          (bin_summary[1][drop_idx] + bin_summary[1][drop_idx - 1]))
        y_tmp_lower[drop_idx * 2 - 1] = new_prob_lower  # New value
        y_tmp_lower[drop_idx * 2 - 2] = new_prob_lower  # Same value
        # Test upper:
        x_tmp_upper = np.array(x)
        x_tmp_upper = np.delete(x, drop_idx * 2 + 2)  # Lower boundary of larger bin
        x_tmp_upper = np.delete(x_tmp_upper, drop_idx * 2 + 1)  # Upper boundary of *this bin
        y_tmp_upper = np.array(y)
        new_prob_upper = ((bin_summary[1][drop_idx] * bin_summary[0][drop_idx] +
                          bin_summary[1][drop_idx + 1] * bin_summary[0][drop_idx + 1]) /
                          (bin_summary[1][drop_idx] + bin_summary[1][drop_idx + 1]))
        y_tmp_upper[drop_idx * 2] = new_prob_upper  # New value, bin guaranteed to exist.
        try:  # Bin doesn't exist if it is last
            y_tmp_upper[drop_idx * 2 + 1] = new_prob_upper  # New value
        except IndexError:
            pass
        # Now, which bin to add it to?
        # Compare the two:
        int_mod_lower = interp1d(x_tmp_lower, y_tmp_lower, bounds_error=False)
        int_mod_upper = interp1d(x_tmp_upper, y_tmp_upper, bounds_error=False)
        int_mod_lower._fill_value_below = 0
        int_mod_lower._fill_value_above = 1
        int_mod_upper._fill_value_below = 0
        int_mod_upper._fill_value_above = 1
        # Left (smaller) bin: idx
        score_lower = estimate_performance(int_mod_lower, data_class, data_scores)
        score_upper = estimate_performance(int_mod_upper, data_class, data_scores)
        if((score_lower['auc_roc'] > score_upper['auc_roc'] and merge_criterion == 'auc_roc') or
           (score_lower['mse'] < score_upper['mse'] and merge_criterion == 'mse')):
            # Select the model with better auc_roc.
            x = x_tmp_lower
            y = y_tmp_lower
            bin_summary[1][drop_idx - 1] = bin_summary[1][drop_idx] + bin_summary[1][drop_idx - 1]
            bin_summary[0][drop_idx - 1] = new_prob_lower
            tmp = credible_interval(k=round(bin_summary[0][drop_idx - 1] * bin_summary[1][drop_idx - 1]),
                                    n=(bin_summary[1][drop_idx - 1]))
            width_of_intervals[drop_idx - 1] = tmp['p_max'] - tmp['p_min']
            credible_intervals.pop(drop_idx)
            credible_intervals[drop_idx - 1] = tmp
            int_mod = int_mod_lower
        else:
            x = x_tmp_upper
            y = y_tmp_upper
            bin_summary[1][drop_idx + 1] = bin_summary[1][drop_idx] + bin_summary[1][drop_idx + 1]
            bin_summary[0][drop_idx + 1] = new_prob_upper
            tmp = credible_interval(k=round(bin_summary[0][drop_idx + 1] * bin_summary[1][drop_idx + 1]),
                                    n=(bin_summary[1][drop_idx + 1]))
            width_of_intervals[drop_idx + 1] = tmp['p_max'] - tmp['p_min']
            credible_intervals[drop_idx + 1] = tmp
            credible_intervals.pop(drop_idx)
            int_mod = int_mod_upper
    width_of_intervals = np.delete(width_of_intervals, drop_idx)
    # Drop samples from bin_summary[1][drop_idx]. The samples are previously added to adequate new bin.
    bin_summary = (np.delete(bin_summary[0], drop_idx), np.delete(bin_summary[1], drop_idx))
    updated_rcir_model = {'model': int_mod, 'credible level': rcir_model['credible level'],
                          'credible intervals': credible_intervals, 'width of intervals': width_of_intervals,
                          'bin summary': bin_summary, 'd': rcir_model['d']}
    # Create new rcir_model object and return. (i.e. stich together new pieces of info.)
    return(updated_rcir_model)


def predict_rcir(rcir_model, data_scores):
    # Function for predicting probabilities given an RCIR-model
    # Uses the interpolation model in the rcir_model dict.
    return(rcir_model['model'](data_scores))


def model_log_likelihood(ir_model, data_scores, data_class, equiv_sample=2):
    # Function for calculating likelihood of binning model given training data.
    # The training data needs to be the same as used for training the classifier.
    # Can this approach be extended to use testing set instead? It would be needed
    # as the target is bootstrapped isotonic regression.
    # Also, IR produces a binning with _gaps_!
    log_likelihood = 0
    probabilities = predict(ir_model, data_scores)
    unique_probabilities = np.unique(probabilities)
    # bin_alphas = []
    # n_bins = float(len(unique_probabilities))
    for p in unique_probabilities:
        pbj = max(5e-3, min(1 - 5e-3, p))
        samples = float(sum(p == probabilities))  # Count of samples falling into *this bin
        alpha = pbj * equiv_sample  # WHAT ABOUT THE BIN WHERE p = 0 ?
        beta = (1 - pbj) * equiv_sample
        n_negative = sum(data_class[probabilities == p] == 0)
        n_positive = sum(data_class[probabilities == p] == 1)
        log_likelihood += gammaln(equiv_sample) - gammaln(samples + equiv_sample)
        log_likelihood += gammaln(n_positive + alpha) - gammaln(alpha)
        log_likelihood += gammaln(n_negative + beta) - gammaln(beta)
    # It doesn't really matter what we normalize this with, i.e. every model log-likelihood
    # can be reduced by max(log-likelihood) (or min). It corresponds to division in euclidean
    # space, i.e. normalization.
    return(log_likelihood)


def relative_log_likelihood(model, training_scores, training_class, test_scores,
                            test_class, y_min=1e-3, y_max=1 - 1e-3):
    # Estimate model log-likelihood directly using Bernoulli distributions.
    # P(score) is identical for all cases in one testing set (and cannot be estimated),
    # hence it can be and is ignored.
    # The IR model is modified to contain no gaps with interpolation, i.e.
    # all testing data points are mapped into original bins enabling us to
    # estimate the likelihood of the data using beta-distributions.
    # Will we run into the same problem with beta-distributions not
    # properly taking neighboring bins into account as priors? I.e.
    # estimating erroneous, too conservative probabilities for data.
    # (This problem should then exist in the formulation utilizing
    # gamma-functions as well.)
    # Proportion between output on two dataset on this should be same as the
    # proportion of the output of model_log_likelihood.
    # "Uniform prior" (beta(1, 1)) assumed for all bins.
    training_prob = predict(model, training_scores)
    unique_probabilities = np.unique(training_prob)
    log_likelihood = 0  # Set initial value.
    # For tests only:
    # equiv_sample = 2  # Does this setup apply for non-fixed sized bins (e.g. IR)?
    # Change model so that all probabilities map to predefined values,
    # i.e. remove gaps from interpolation model. The results is a piecewise constant
    # function.
    x_y_values = correct_for_point_bins(model.f_.x, model.f_.y)
    new_int_mod = interp1d(x_y_values['x'], x_y_values['y'])
    modified_model = modify_model(new_int_mod)
    test_prob = predict(modified_model, test_scores)
    test_sum = 0
    for p in unique_probabilities:
        # Estimate parameters for bin-related beta-distributions:
        n_alpha = sum(training_class[training_prob == p] == 1)  # Add priors?
        n_beta = sum(training_class[training_prob == p] == 0)
        if(n_alpha + n_beta < 1):
            print("Some bug? No positive or negative training samples for *this probability")
            print(p)
        # Parameters for testing set:
        n_positive = sum(test_class[test_prob == p] == 1)
        n_negative = sum(test_class[test_prob == p] == 0)
        log_likelihood += n_positive * np.log(p)
        log_likelihood += n_negative * np.log(1 - p)
        test_sum += n_positive + n_negative
    # The loop above should go through all datapoints once. Check.
    if(test_sum != test_class.shape[0]):
        print("All samples not counted once. Likelihoods unreliable!")
        print(test_sum)
        print(test_class.shape[0])
        # As the probability of any given point is usually 1% or less, we cannot
        # possibly compare models that leave out datapoints!
    return(log_likelihood)


def train_wabir(data_class, data_scores, sampling_rate=.95, n_models=200, y_min=1e-3, y_max=1 - 1e-3):
    # Function for training an ensemble of isotonic regression models
    models = []
    n_samples = data_class.shape[0]
    for i in range(n_models):
        # print("Working on {0} of {1} models.".format(i, n_models))
        tmp_model = IsotonicRegression(y_min=y_min, y_max=y_max, out_of_bounds='clip')
        idx = np.random.randint(low=0, high=n_samples, size=int(sampling_rate * n_samples))
        tmp_data_class = data_class[idx]
        tmp_data_scores = data_scores[idx]
        tmp_model.fit(X=tmp_data_scores, y=tmp_data_class)
        # We might need 'model_log_likelihood' to get numerically treatable likelihoods!
        # model_log_likelihoods with gamma-functions does not seem to generalize to testing set!!!
        # tmp_model_log_likelihood = model_log_likelihood(tmp_model, tmp_data_scores, tmp_data_class)
        # CHANGE THE ROW BELOW TO USE ALL DATA FOR LIKELIHOOD ESTIMATION
        tmp_log_likelihood = relative_log_likelihood(tmp_model, tmp_data_scores, tmp_data_class, data_scores, data_class)
        # If we store both of the above, then normalize so that e.g. max model likelihood
        # equals zero, then the remaining 'model_log_likelihoods' and 'log_likelihoods'
        # should be identical (save for priors?).
        models.append({'model': tmp_model, 'log_likelihood': tmp_log_likelihood})  # , 'model_log_likelihood': tmp_model_log_likelihood})
    max_log_likelihood = max([item['log_likelihood'] for item in models])
    # max_model_log_likelihood = max([item['model_log_likelihood'] for item in models])
    i = 1
    for item in models:  # Standardize relative log-likelihood to something usable:
        i += 1
        # Hmm, the likelihoods vary very much. Should something be done about it?
        item['log_likelihood'] -= max_log_likelihood + 5
        # item['model_log_likelihood'] -= max_model_log_likelihood
    return(models)


def predict_wabir(models, data_scores, weighted_average=True):
    # Function for predicting from ensemble of models
    # Weighted average:
    if(weighted_average):
        normalizing_factor = sum(np.exp([item['log_likelihood'] for item in models]))
        probabilities = np.sum([item['model'].predict(data_scores) *
                                np.exp(item['log_likelihood']) / normalizing_factor for
                                item in models], axis=0)
    else:
        probabilities = np.mean([item['model'].predict(data_scores) for item in models], axis=0)
    return(probabilities)


def get_metrics(data_class, data_prob, k=100):
    mse = mean_squared_error(data_class, data_prob)
    auc_roc = roc_auc_score(data_class, data_prob)
    ece = expected_calibration_error(data_class, data_prob, k=k)
    mce = maximum_calibration_error(data_class, data_prob, k=k)
    max_p = max(data_prob)
    results = {'mse': mse, 'auc-roc': auc_roc, 'ece': ece, 'mce': mce, 'max_p': max_p}
    return(results)


def average_metrics(metrics):
    # Function for printing AVERAGE of list of metrics produced by get_metrics.
    mse = np.mean([item['mse'] for item in metrics])
    auc_roc = np.mean([item['auc-roc'] for item in metrics])
    ece = np.mean([item['ece'] for item in metrics])
    mce = np.mean([item['mce'] for item in metrics])
    max_p = np.mean([item['max_p'] for item in metrics])
    return({'mse': mse, 'auc_roc': auc_roc, 'ece': ece, 'mce': mce, 'max_p': max_p})


def metrics_for_high_scoring_samples(test_class, bbq_prob, other_prob):
    # Function for calculating metrics for predictions that map to
    # higher probabilities than max(bbq_prob)
    # Due to monotonicity of IR, BIR, and WABIR, these samples equals
    # the highest scoring ones.
    max_bbq = max(bbq_prob)
    idx = other_prob > max_bbq
    n_samples = sum(idx)
    n_positive = sum(test_class[idx])
    if n_samples > 0:
        empiric_frequency = n_positive / float(n_samples)
        other_estimate = np.mean(other_prob[idx])
        bbq_estimate = np.mean(bbq_prob[idx])
    else:
        empiric_frequency = None
        other_estimate = None
        bbq_estimate = None
    return({'samples': n_samples, 'empiric frequency': empiric_frequency,
            'estimate': other_estimate, 'bbq estimate': bbq_estimate})


def average_high_scoring(metrics):
    samples = np.mean([item['samples'] for item in metrics])
    empiric_frequency = np.mean([item['empiric frequency'] for item in metrics if item['empiric frequency'] is not None])
    estimate = np.mean([item['estimate'] for item in metrics if item['estimate'] is not None])
    bbq_estimate = np.mean([item['bbq estimate'] for item in metrics if item['bbq estimate'] is not None])
    return([samples, empiric_frequency, estimate, bbq_estimate])


def metrics_at(data_class, data_prob, data_scores, low=.99, high=1.00, k=100):
    # By default, this function returns metrics for the top 1% scoring samples.
    # The function is designed to return metrics for a slice of the samples,
    # where ordering is defined by score.
    # NOTE: AUC-ROC IS NOT A CORRECT METRIC HERE. THE 'AUC-ROC' MEASURED IS
    # SIMPLY THE AUC-ROC IN SPECIFIED FIXED-WIDTH BIN!!
    idx = np.argsort(data_scores)
    top_idx = idx[int(low * len(idx)):int(high * len(idx))]
    test_class = data_class[top_idx]
    test_prob = data_prob[top_idx]
    metrics = get_metrics(test_class, test_prob, k=k)
    n_samples = len(top_idx)
    n_positive = sum(data_class[top_idx])
    if n_samples > 0:
        empiric_frequency = n_positive / float(n_samples)
        frequency_estimate = np.mean(data_prob[top_idx])
    else:
        empiric_frequency = None
        frequency_estimate = None
    metrics['empiric frequency'] = empiric_frequency
    metrics['frequency estimate'] = frequency_estimate
    return(metrics)


def all_metrics_at(data_class, data_prob, data_scores):
    metrics = []
    for i in range(90, 100):
        metrics.append(metrics_at(data_class, data_prob, data_scores, low=float(i * .01),
                                  high=float((i + 1) * .01)))
    return(metrics)


def print_at_metrics(at_metrics, level):
    # NOTE: THE WAY WE HAVE USED 'LEVEL' HERE AS A FIXED SIZE LIST WITH PREDEFINED
    # VALUES WILL MAKE THIS CRASH IF CHANGES ARE MADE!
    types = ['ir', 'bir', 'wabir', 'rcir40', 'rcir30', 'rcir20', 'rcir10', 'rcir05', 'bbq', 'enir']
    print("At " + str(level))
    print("\tMSE(?) \t\tAUC-ROC \tECE \t\tmax(p) \t\tFrequency \tProbability")
    levels = [.95, .96, .97, .98, .99]
    idx = levels.index(level)  # Match.
    for key in types:  # Enforces order instead of using at_metrics.keys()
        tmp_1 = at_metrics[key]  # A list of dicts extracted from a dict.
        tmp = {}  # Store results here.
        # Average over all dicts:
        for key_2 in tmp_1[0].keys():  # First list contains all keys
            # Pick every fifth item from list (five items in levels!)
            tmp[key_2] = np.average([item[key_2] for item in tmp_1[idx::len(levels)]])
        print("{0} \t{1:>9.7} \t{2:>9.7} \t{3:>9.7} \t{4:>9.7} \t{5:>9.7} \t{6:>9.7}".format(key,
              tmp['mse'], tmp['auc-roc'], tmp['ece'], tmp['max_p'],
              tmp['empiric frequency'], tmp['frequency estimate']))


def plot_metrics_at(data_class, bbq_prob, other_prob, data_scores, metric='mse', file_name='metrics_at.png'):
    # Function for plotting metrics at.
    bbq_metrics = all_metrics_at(data_class, bbq_prob, data_scores)
    tmp_bbq = [item[metric] for item in bbq_metrics]
    other_metrics = all_metrics_at(data_class, other_prob, data_scores)
    tmp_other = [item[metric] for item in other_metrics]
    idx = [item * .01 for item in range(91, 101)]
    plt.plot(idx, tmp_bbq)
    plt.plot(idx, tmp_other)
    plt.title("Metrics for diffferently scoring samples.")
    plt.savefig(file_name)
    plt.gcf().clear()


def plot_calibration_mapping(calibration_model, min_score, max_score, resolution=1000,
                             file_name='calibration_mapping.png'):
    """
    Function for plotting what probabilities different scores get mapped to.
    "General purpose prediction function"
    Perhaps add probability distribution of training data (or testing?)?
    Would indicate how many samples fall into one bin.
    """
    diff = max_score - min_score
    scores = [min_score + i * diff / float(resolution) for i in range(resolution + 1)]
    try:  # IR model
        probabilities = calibration_model.predict(scores)
    except:
        try:  # ENIR
            import rpy2.robjects as robjects
            from rpy2.robjects.packages import importr
            enir = importr('enir')
            r = robjects.r
            # Automatic conversion or numpy arrays to R-vectors
            import rpy2.robjects.numpy2ri
            rpy2.robjects.numpy2ri.activate()
            # ENIR-MODEL MIGHT NEED TO BE PUSHED TO R-ENVIRONMENT?
            probabilities = enir.enir_predict(calibration_model, robjects.FloatVector(scores))
            probabilities = np.array(probabilities)
        except:
            try:  # BBQ
                from oct2py import octave
                octave.eval("addpath('./calibration/BBQ/')", verbose=False)
                octave.push('scores', scores, verbose=False)
                octave.push('calibration_model', calibration_model, verbose=False)
                octave.eval('probabilities = predict(calibration_model, scores, 1)', verbose=False)
                probabilities = octave.pull('probabilities', verbose=False)
                probabilities = np.array([item[0] for item in probabilities])
            except:
                pass  # Continue with BIR and WABIR? RCIR?
    # Plot score vs. probability:
    plt.plot(scores, probabilities)
    plt.title("Calibration mapping")
    plt.savefig(file_name)
    plt.gcf().clear()

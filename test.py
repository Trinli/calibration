# Test script for reliably calibrated isotonic regression.
# This script contains tests for three datasets.
# In the basic setting, we find the isotonic regression model,
# the 'best' RCIR-model using validation set performance, and
# compare the testing set performance of these to a Naeini model trained with
# both the training and validation set. This way both models have access to the
# same data. The RCIR-model could utilize the data more efficiently by cross-validation
# but that is left out here.
# ARGH. IR MODEL IS NOT DIRECTLY COMPARABLE. SHOULD USE THE SAME DATA AS THE
# NAEINI MODEL AS NO CV IS REQUIRED.
# The script can also be used to find the RCIR-model that corresponds to the 'd' value
# of a Naeini model. In this case, both models are trained on the same data as the
# RCIR does not need a validation set in the context. Performance is then compared
# on a separate testing set.

import isotonic
from sklearn.isotonic import IsotonicRegression
# from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from scipy.interpolate import interp1d
from oct2py import octave
from sklearn.metrics import roc_auc_score
# Enable octave to find Naeini's functions!
# (download from https://github.com/pakdaman/calibration)
octave.eval("addpath('./calibration/BBQ/')", verbose=False)

# Set test parameters:
dataset = input("Select dataset to run experiment on (1, 2, or 3): ")
n_iterations = input("Set number of iterations (30 used in paper): ")
# metric = input("Select metric ('mse' or 'auc_roc'): ")  # Perhaps allow only auc-roc and exclude mse?
metric = 'auc_roc'  # Can also be set to 'mse' to run the algorithm with mse-based bin merges.
# reshuffle = input("Shuffle data? (y/n): ")  # Should be shuffled at least once. Perhaps remove option.
# It seems that there is no convenient way of estimating the credible intervals, not to speak of the maximum
# credible intervals for the Naeini procedure. Hence the 'Naeini vs. RCIR with d set by Naeini vs. IR'
# is pointless.
# model_comparison = input("Naeini vs. RCIR-CV ('Naeini vs. RCIR-CV') or Naeini vs. RCIR? ")
model_comparison = 'Naeini vs. RCIR-CV'  # Set to anything else, it will run Naeini vs. RCIR with d set by the Naeini model.
# model_comparison = 'Naeini vs. RCIR with d set by Naeini vs. IR'
naeini_metrics = []
rcir_better_than_naeini_metrics = []
rcir_cv_best_model_metrics = []
s_isotonic_regression_metrics = []
threshold = 0.10
t_isotonic_metrics = []

# Load dataset:
if dataset == 1:
    # Read data for test 1:
    test_description = "test1"
    data_class = isotonic.load_pickle("./data/dataset_1_class.pickle")
    data_scores = isotonic.load_pickle("./data/dataset_1_scores.pickle")
elif dataset == 2:
    # Read data for test 2:
    test_description = "test2"
    data_class = isotonic.load_pickle("./data/dataset_2_class.pickle")
    data_scores = isotonic.load_pickle("./data/dataset_2_scores.pickle")
elif dataset == 3:
    test_description = "test3"
    data_class = isotonic.load_pickle("./data/dataset_3_class.pickle")
    data_scores = isotonic.load_pickle("./data/dataset_3_scores.pickle")
elif dataset == 4:  # Hidden experiment. Does not really provide anything interesting.
    test_description = "test4"
    data_class = np.random.binomial(1, .5, 30000)
    data_scores = np.random.uniform(low=0, high=1, size=30000)
elif dataset == 5:  # Test mode
    tmp_class = isotonic.load_pickle("./data/dataset_1_class.pickle")
    tmp_scores = isotonic.load_pickle("./data/dataset_1_scores.pickle")
    test_description = "test1"
    data_class = tmp_class[:1000]
    data_scores = tmp_scores[:1000]

else:
    print("Not a valid dataset selection.")
    import sys
    sys.exit()
print("Dataset with " + str(data_class.shape[0]) + " samples loaded.")

if(metric == 'mse' or metric == 'auc_roc'):
    pass  # All in order.
else:
    print("Not a valid metric selection.")
    import sys
    sys.exit()

for j in range(n_iterations):
    # Shuffle samples:
    n_rows = data_scores.shape[0]
    idx = np.random.permutation(range(n_rows))
    data_class = data_class[idx]
    data_scores = data_scores[idx]
    # If we compare Naeini's model to RCIR with validation set, we need to allow Naeini's model
    # to use the data used for validation with the RCIR for training Naeini's model. It does not
    # need a validation sample, whereas RCIR with CV does.
    # If we want to compare Naeini's model to RCIR with d set by Naeini's model, then RCIR does
    # not either need any validation set, and the same data should be used for training both
    # models.
    # Data for RCIR with CV, or RCIR with d provided by Naeini's model
    test_class = data_class[:n_rows * 1 // 4]
    training_class = data_class[n_rows // 4: 3 * n_rows // 4]
    validation_class = data_class[3 * n_rows // 4:]
    test_scores = data_scores[:n_rows * 1 // 4]
    training_scores = data_scores[n_rows // 4: 3 * n_rows // 4]
    validation_scores = data_scores[3 * n_rows // 4:]
    # Give all models exactly same data, save for validation set for IR
    naeini_training_class = training_class
    naeini_training_scores = training_scores
    # if(model_comparison == 'Naeini vs. RCIR-CV'):
    #     # Use training set + validation set for Naeini model as training set.
    #     naeini_training_class = data_class[n_rows * 1 // 4:]
    #     naeini_training_scores = data_scores[n_rows * 1 // 4:]
    # else:
    #     # Naeini vs. RCIR (d set by Naeini)
    #     naeini_training_scores = training_scores
    #     naeini_training_class = training_class

    # Standard isotonic regression. Needs to have same data as Naeini to produce fair metrics.
    s_isotonic_regression = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    s_isotonic_regression.fit(X=training_scores, y=training_class)
    s_metrics = isotonic.estimate_performance(s_isotonic_regression, test_class, test_scores)
    s_training_prob = s_isotonic_regression.predict(training_scores)
    s_max_p = max(s_training_prob)
    s_data = np.unique(s_training_prob, return_counts=True)
    s_credible_intervals = [isotonic.credible_interval(np.round(p * n), n) for
                            (p, n) in zip(s_data[0], s_data[1])]
    s_width_of_intervals = np.array([row['p_max'] - row['p_min'] for row in s_credible_intervals])
    s_max_width = max(s_width_of_intervals)
    s_bins = len(np.unique(s_training_prob))
    s_test_prob = s_isotonic_regression.predict(test_scores)
    s_mce = isotonic.maximum_calibration_error(test_class, s_test_prob)
    s_ece = isotonic.expected_calibration_error(test_class, s_test_prob)
    s_acc = isotonic.accuracy(test_class, s_test_prob)
    s_isotonic_regression_metrics.append({'mse': s_metrics['mse'],
                                          'auc_roc': s_metrics['auc_roc'],
                                          'max_width': s_max_width,
                                          'bins': s_bins,
                                          'max_p': s_max_p,
                                          'mce': s_mce,
                                          'ece': s_ece,
                                          'acc': s_acc})

    # Train separate isotonic regression model model for RCIR. It need a separate validation
    # set and must thus be different from the s_isotonic_regression above.
    # Usually I would use y_min=.0001, and y_max=.99. Probabilities of 0 or 1 are not
    # plausible and can cause problems downstream.
    isotonic_regression_model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')

    # Fit isotonic regression model
    isotonic_regression_model.fit(X=training_scores, y=training_class)
    training_probabilities = isotonic_regression_model.predict(T=training_scores)
    print("Isotonic regression model trained.")

    #  To evaluate goodness of fit, we need to predict the probabilities for the test scores
    #  using our isotonic regression model (here stored in an interpolation model) and
    #  calculate some test statistics for it. Let it be mse and AUC-ROC.

    # Line below corrects for bin widths (i.e. not use interpolation
    # between bins). Will produce equivalent results for training data
    # but slightly different for testing data.
    # interpolation_model = isotonic.modify_model(model_tmp)  # This is not how IR usually works.
    # IsotonicRegression uses an interpolation model at its core. The 'x' and 'y' define this model
    # completely. In some rare cases, the bins end up having zero width and the library
    # IsotonicRegression produces x and y items for the interpolation model that don't have
    # pairs. The rest of our code is based on the assumption that the pair exist, hence we make
    # a check and correction if necessary.
    tmp_x = isotonic_regression_model.f_.x
    tmp_y = isotonic_regression_model.f_.y
    tmp = isotonic.correct_for_point_bins(tmp_x, tmp_y)
    x = tmp['x']
    y = tmp['y']

    # The following array contains all information defining the IR transformation
    data = np.unique(training_probabilities, return_counts=True)
    credible_intervals = [isotonic.credible_interval(np.round(p * n), n) for (p, n) in zip(data[0], data[1])]
    width_of_intervals = np.array([row['p_max'] - row['p_min'] for row in credible_intervals])

    # print "Test- and validation set performance, (standard) isotonic regression"  # All bins.
    isotonic_regression_performance = isotonic.estimate_performance(isotonic_regression_model, test_class, test_scores)
    # print(isotonic_regression_performance)
    # print(isotonic.estimate_performance(interpolation_model, training_class, training_scores))
    # print(isotonic.estimate_performance(isotonic_regression_model, validation_class, validation_scores))
    isotonic.plot_intervals(data, credible_intervals,
                            file_name="./" + test_description + "/all_bins.png",
                            label="Credible intervals")

    # Create reliability diagram:
    isotonic.plot_reliability_diagram(isotonic_regression_model, test_scores, test_class,
                                      file_name="./" + test_description + "/reliability_diagram" + ".png",
                                      label="Reliability diagram")

    # Naeini's model
    # Comparison to Naeini's model (it's a matlab model, using Octave).
    print("Training Naeini-model. This might take a while (~20 minutes on a laptop).")
    octave.push('training_scores', naeini_training_scores, verbose=False)
    octave.push('training_class', naeini_training_class, verbose=False)
    octave.eval("options.N0 = 2", verbose=False)
    octave.eval("BBQ_model = build(training_scores', training_class', options)", verbose=False)
    # In the following, '1' indicates model averaging, as done in the paper by Naeini & al.
    octave.eval("training_bbq_prob = predict(BBQ_model, training_scores, 1)", verbose=False)
    training_bbq_prob = octave.pull("training_bbq_prob", verbose=False)
    training_bbq_prob = np.array([item[0] for item in training_bbq_prob])
    octave.push('test_scores', test_scores, verbose=False)
    octave.eval("test_bbq_prob = predict(BBQ_model, test_scores, 0)", verbose=False)
    test_bbq_prob = octave.pull("test_bbq_prob", verbose=False)
    test_bbq_prob = np.array([item[0] for item in test_bbq_prob])
    naeini_bins = len(np.unique(training_bbq_prob))
    naeini_mse = sum((test_bbq_prob - test_class) ** 2) / len(test_bbq_prob)
    naeini_auc_roc = roc_auc_score(test_class, test_bbq_prob)
    samples_per_bin = int(len(naeini_training_class) / naeini_bins)
    naeini_p = max(training_bbq_prob)  # Highest predicted value.
    # Estimate credible intervals. np.unique() returns a list of two lists. The first one contains probabilities,
    # the second counts. Their product corresponds to 'k', whereas the counts correspond to 'n'.
    # For some reason, the library produces some negative values. These are set to zero.
    naeini_data = np.unique(training_bbq_prob, return_counts=True)
    # The naeini-model produces negative probabilities for some reason (?!?)
    # Clear out the 'naeini_corrected data'. Turns out it was just my error. Doesn't affect results, though.
    naeini_corrected_data = [[0 if item < 0 else item for item in naeini_data[0]], naeini_data[1]]
    naeini_credible_intervals = [isotonic.credible_interval(np.round(p * n), n) for
                                 (p, n) in zip(naeini_corrected_data[0], naeini_corrected_data[1])]
    naeini_width_of_intervals = np.array([row['p_max'] - row['p_min'] for row in naeini_credible_intervals])
    naeini_max_width = max(naeini_width_of_intervals)
    # naeini_mce = isotonic.maximum_calibration_error(test_bbq_prob, test_class)
    # naeini_ece = isotonic.expected_calibration_error(test_bbq_prob, test_class)
    naeini_mce = isotonic.maximum_calibration_error(test_class, test_bbq_prob)
    naeini_ece = isotonic.expected_calibration_error(test_class, test_bbq_prob)
    naeini_acc = isotonic.accuracy(test_class, test_bbq_prob)
    naeini_metrics.append({'mse': naeini_mse, 'auc_roc': naeini_auc_roc, 'max_width': naeini_max_width,
                           'bins': naeini_bins, 'samples_per_bin': samples_per_bin, 'max_p': naeini_p,
                           'mce': naeini_mce, 'ece': naeini_ece, 'acc': naeini_acc})
    naeini_flag = True  # Flag for tracking whether the new RCIR model is the first one beating the naeini model.
    print("Naeini model trained.")
    print(naeini_metrics[j])

    # RCIR experiment
    i = 0  # Tracker for number of bin merges.
    best_score = {'mse': 1.0, 'auc_roc': 0.0}  # Set to worst possible, used in comparison later.
    isotonic_regression_max_width = max(width_of_intervals)  # Store largest 'd'
    first_model_below_threshold = True  # Reset flag.
    while(len(width_of_intervals) > 2):  # Something has length... can still drop bins
        i += 1  # Add one to counter.
        max_interval = max(width_of_intervals)
        # print "Max width: " + str(max_interval)
        drop_idx = width_of_intervals.tolist().index(max_interval)
        if drop_idx == 0:  # Exception handling. Fist bin has largest credible interval.
            # remove first two elements in x and y, update new first elements,
            # remove first element from width_of_intervals
            # and remove first elements from data[0] and data[1]
            y = np.delete(y, [0, 1])  # Drop first and second items.
            x = np.delete(x, [0, 1])
            new_prob = (data[0][0] * data[1][0] + data[0][1] * data[1][1]) / (data[1][0] + data[1][1])
            y[0] = new_prob
            try:  # y[1] doesn't exist if this is also the last bin.
                y[1] = new_prob
            except IndexError:
                pass
            # Leave x as is. data and width_of_intervals handled at end of loop.
            int_mod = interp1d(x, y, bounds_error=False)
            int_mod._fill_value_below = 0
            int_mod._fill_value_above = 1
            # print("Test-, and training performance, " + str(i) + " bins removed.d")
            # print(isotonic.estimate_performance(int_mod, test_class, test_scores))
            # print(isotonic.estimate_performance(int_mod, training_class, training_scores))
            tmp = isotonic.credible_interval(k=round(data[0][0] * data[1][0] + data[0][1] * data[1][1]),
                                             n=(data[1][0] + data[1][1]))
            width_of_intervals[0] = tmp['p_max'] - tmp['p_min']
            credible_intervals.pop(drop_idx)  # Remove line from credible intervals
            credible_intervals[0] = tmp
            data[0][1] = new_prob
            data[1][1] = data[1][0] + data[1][1]
            validation_score = isotonic.estimate_performance(int_mod, validation_class, validation_scores)
            # print(isotonic.estimate_performance(int_mod, test_class, test_scores))
            # print(isotonic.estimate_performance(int_mod, training_class, training_scores))
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
            new_prob = (data[0][-1] * data[1][-1] + data[0][-2] * data[1][-2]) / (data[1][-1] + data[1][-2])
            # Hmm, there might be two bins for this y
            if(two_y_end):
                y[-2] = new_prob
            y[-1] = new_prob
            tmp = isotonic.credible_interval(k=round(data[0][-1] * data[1][-1] + data[0][-2] * data[1][-2]),
                                             n=(data[1][-1] + data[1][-2]))
            width_of_intervals[-2] = tmp['p_max'] - tmp['p_min']
            credible_intervals.pop(-1)  # Drop last.
            credible_intervals[-1] = tmp
            data[0][-2] = new_prob
            data[1][-2] = data[1][-1] + data[1][-2]
            # if((drop_idx != len(width_of_intervals)) or (drop_idx != 0):  # Main handling
            int_mod = interp1d(x, y, bounds_error=False)
            int_mod._fill_value_below = 0
            int_mod._fill_value_above = 1
            # print("Testing set performance, " + str(i) + " bins removed.c")
            # print(isotonic.estimate_performance(int_mod, test_class, test_scores))
            validation_score = isotonic.estimate_performance(int_mod, validation_class, validation_scores)
        else:
            # Main method, i.e. when we are not dealing with the first or last bin.
            # y contains the probability to be dropped twice
            y = np.delete(y, drop_idx * 2 + 1)
            y = np.delete(y, drop_idx * 2)
            # Test lower:
            x_tmp_lower = np.array(x)  # Create NEW array!!
            x_tmp_lower = np.delete(x_tmp_lower, drop_idx * 2)  # Lower boundary of *this bin
            x_tmp_lower = np.delete(x_tmp_lower, drop_idx * 2 - 1)  # Upper boundary of smaller bin
            y_tmp_lower = np.array(y)  # Creates _new_ array!!!
            new_prob_lower = ((data[1][drop_idx] * data[0][drop_idx] +
                              data[1][drop_idx - 1] * data[0][drop_idx - 1]) /
                              (data[1][drop_idx] + data[1][drop_idx - 1]))
            y_tmp_lower[drop_idx * 2 - 1] = new_prob_lower  # New value
            y_tmp_lower[drop_idx * 2 - 2] = new_prob_lower  # Same value
            # Test upper:
            x_tmp_upper = np.array(x)
            x_tmp_upper = np.delete(x, drop_idx * 2 + 2)  # Lower boundary of larger bin
            x_tmp_upper = np.delete(x_tmp_upper, drop_idx * 2 + 1)  # Upper boundary of *this bin
            y_tmp_upper = np.array(y)
            new_prob_upper = ((data[1][drop_idx] * data[0][drop_idx] +
                              data[1][drop_idx + 1] * data[0][drop_idx + 1]) /
                              (data[1][drop_idx] + data[1][drop_idx + 1]))
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
            score_lower = isotonic.estimate_performance(int_mod_lower, validation_class, validation_scores)
            score_upper = isotonic.estimate_performance(int_mod_upper, validation_class, validation_scores)
            if((score_lower['auc_roc'] > score_upper['auc_roc'] and metric == 'auc_roc') or
               (score_lower['mse'] < score_upper['mse'] and metric == 'mse')):  # Select the model with better auc_roc.
                x = x_tmp_lower
                y = y_tmp_lower
                data[1][drop_idx - 1] = data[1][drop_idx] + data[1][drop_idx - 1]
                data[0][drop_idx - 1] = new_prob_lower
                tmp = isotonic.credible_interval(k=round(data[0][drop_idx - 1] * data[1][drop_idx - 1]),
                                                 n=(data[1][drop_idx - 1]))
                width_of_intervals[drop_idx - 1] = tmp['p_max'] - tmp['p_min']
                credible_intervals.pop(drop_idx)
                credible_intervals[drop_idx - 1] = tmp
                int_mod = int_mod_lower
                # print("Testing set performance, " + str(i) + " bins removed.b")
                # print(isotonic.estimate_performance(int_mod_lower, test_class, test_scores))
            else:
                x = x_tmp_upper
                y = y_tmp_upper
                data[1][drop_idx + 1] = data[1][drop_idx] + data[1][drop_idx + 1]
                data[0][drop_idx + 1] = new_prob_upper
                tmp = isotonic.credible_interval(k=round(data[0][drop_idx + 1] * data[1][drop_idx + 1]),
                                                 n=(data[1][drop_idx + 1]))
                width_of_intervals[drop_idx + 1] = tmp['p_max'] - tmp['p_min']
                credible_intervals[drop_idx + 1] = tmp
                credible_intervals.pop(drop_idx)
                int_mod = int_mod_upper
                # print("Testing set performance, " + str(i) + " bins removed.a")
                # print(isotonic.estimate_performance(int_mod_upper, test_class, test_scores))
                # print(isotonic.estimate_performance(int_mod_upper, training_class, training_scores))
            validation_score = isotonic.estimate_performance(int_mod, validation_class, validation_scores)
        width_of_intervals = np.delete(width_of_intervals, drop_idx)
        # Drop samples from data[1][drop_idx]. The samples are previously added to adequate new bin.
        data = (np.delete(data[0], drop_idx), np.delete(data[1], drop_idx))
        isotonic.plot_intervals(data, credible_intervals,
                                file_name="./" + test_description + "/" + str(i) + "_bins_dropped.png",
                                label="Credible intervals, d=" +
                                str(round(max(width_of_intervals), 3)))
        isotonic.plot_reliability_diagram(int_mod, test_scores, test_class,
                                          label="Reliability diagram, " + str(i) + " bins dropped",
                                          file_name="./" + test_description + "/reliability_diagram" + str(i) + ".png")
        # Store best model:
        # if(validation_score['auc_roc'] > best_score['auc_roc']):
        if((validation_score['auc_roc'] > best_score['auc_roc'] and metric == 'auc_roc') or
           (validation_score['mse'] < best_score['mse'] and metric == 'mse')):  # Select the model with better auc_roc.
            # This is the validation set control
            best_model = int_mod  # This is the RCIR model (technically an interpolation model)
            best_model_bin_merges = i  # Number of bin merges
            best_model_max_width = max(width_of_intervals)  # Maximum width of credible intervals
            best_model_bins_left = len(width_of_intervals)  # Number of bins left
            best_model_max_p = max(isotonic.predict(best_model, training_scores))
            best_score = validation_score  # Note!! Best score on VALIDATION SET, not testing set.
            best_model_test_prob = isotonic.predict(best_model, test_scores)
            best_model_mce = isotonic.maximum_calibration_error(test_class, best_model_test_prob)
            best_model_ece = isotonic.expected_calibration_error(test_class, best_model_test_prob)
            best_model_acc = isotonic.accuracy(test_class, best_model_test_prob)
        # If the current model is the first one to have smaller maximum credible interval (max_d)
        # than the Naeini-model, then estimate and store metrics for that model.
        # NO POINT IN ANYTHING ELSE THAN RCIR-CV AS COMPARISON FOR NAEINI AS NAEINI DOES NOT REALLY
        # PROVIDE A VALID NUMBER FOR 'd'
        # if(model_comparison != 'Naeini vs. RCIR-CV'):
        #     max_width = max(width_of_intervals)
        #     if(max_width < naeini_max_width and naeini_flag):
        #         naeini_flag = False  # set flag to not enter this block again, store values.
        #         # Testing set metrics of _current_ model:
        #         test_score = isotonic.estimate_performance(int_mod, test_class, test_scores)
        #         max_p = max(isotonic.predict(int_mod, training_scores))  # Maximum value of prediction
        #         bins_left = len(width_of_intervals)
        #         bin_merges = i
        #         rcir_better_than_naeini_metrics.append({'mse': test_score['mse'],
        #                                                 'auc_roc': test_score['auc_roc'],
        #                                                 'max_width': max_width,
        #                                                 'bins': bins_left,
        #                                                 'bin_merges': bin_merges,
        #                                                 'max_p': max_p})
        if(max(width_of_intervals) < threshold and first_model_below_threshold):  # I.e. if this is the first time we fall below the threshold
            first_model_below_threshold = False  # Set flag.
            max_width = max(width_of_intervals)
            # Testing set metrics of _current_ model:
            test_score = isotonic.estimate_performance(int_mod, test_class, test_scores)
            t_test_prob = isotonic.predict(int_mod, test_scores)
            t_max_p = max(isotonic.predict(int_mod, training_scores))  # Maximum value of prediction
            t_bins_left = len(width_of_intervals)
            t_bin_merges = i
            t_mce = isotonic.maximum_calibration_error(test_class, t_test_prob)
            t_ece = isotonic.expected_calibration_error(test_class, t_test_prob)
            t_acc = isotonic.accuracy(test_class, t_test_prob)
            t_isotonic_metrics.append({'mse': test_score['mse'],
                                       'auc_roc': test_score['auc_roc'],
                                       'max_width': max_width,
                                       'bins': t_bins_left,
                                       'bin_merges': t_bin_merges,
                                       'max_p': t_max_p,
                                       'mce': t_mce,
                                       'ece': t_ece,
                                       'acc': t_acc})
    if(model_comparison == 'Naeini vs. RCIR-CV'):
        # At this point, the while-loop has finished. I.e. data and width_of_intervals correspond to a model with
        # _two_ bins remaining.
        best_model_performance = isotonic.estimate_performance(best_model, test_class, test_scores)
        # All values below for the best rcir-model unless explicitly stated otherwise ('_ir').
        rcir_cv_best_model_metrics.append({'mse': best_model_performance['mse'],
                                           'auc_roc': best_model_performance['auc_roc'],
                                           'max_width': best_model_max_width,
                                           'bins': best_model_bins_left,
                                           'bin_merges': best_model_bin_merges,
                                           'max_p': best_model_max_p,
                                           'mse_ir': isotonic_regression_performance['mse'],
                                           'auc_roc_ir': isotonic_regression_performance['auc_roc'],
                                           'max_width_ir': isotonic_regression_max_width,
                                           'mce': best_model_mce,
                                           'ece': best_model_ece,
                                           'acc': best_model_acc})
        print("Best model testing set scores")
        print(rcir_cv_best_model_metrics[j])
        print("Dropped bins " + str(best_model_bin_merges))
    if(model_comparison != 'Naeini vs. RCIR-CV'):
        print("First model to reach Naeini's d: ")
        print(rcir_better_than_naeini_metrics[j])
    print("\n")  # Newline to separate runs

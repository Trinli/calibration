"""
Tests for sample size and sampling rate.
Testing: IR (my version!), ENIR, BEIR, BBQ, and Platt-scaling (ditching single parameter Platt-scaling)
Metrics: MSE, AUC-ROC, ECE, MCE, max(p)

Improvements/changes?
-Heatmap per method for metrics. Color relative to range of all values for different methods?
 Perhaps highlight "best" with sample size / sampling rate -combination.
 Or maybe two heatmaps, one for performance of *this method, and one for *this method
 relative to others (e.g. green for best, red for worst)
-Grid search for testing both sample size and sampling rate at the same time?
-Do we want anything with credible intervals?
-Maybe set k to 10 (instead of 100) to match Naeini's tests.

Notes:
-I'm using the ENIR-version provided by Naeini & al, despite the rare result
 of probabilities mapping to negative values or values higher values than 1.
 Mitigating the problem with a 'min-max'-approach.
-There is also an error in sklearn.isotonic.IsotonicRegression. Using my own code instead.
 I might perhaps publish it.
"""

import my_ir  # Both for IR and BEIR, perhaps change version of IR used by BEIR to make it faster!
import isotonic  # A bunch of functions for estimating goodness metrics. Maybe rename?
import logistic_regression
import datetime
import numpy as np
from oct2py import octave
import matplotlib
# Force matplotlib not to use any Xwindow backend:
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from sklearn.linear_model import LogisticRegression  # Does not support unregularized LR, i.e. Platt-scaling.
# ENIR-code is R:
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
enir = importr('enir')
r = robjects.r
# Automatic conversion or numpy arrays to R-vectors
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
# from sklearn.metrics import roc_auc_score
# Enable octave to find Naeini's functions!
# (download from https://github.com/pakdaman/calibration)
octave.eval("addpath('./calibration/BBQ/')", verbose=False)

# Read in some data
# Set test parameters:
dataset = int(input("Select dataset to run experiment on (1, 2, or 3): "))
n_iterations = int(input("Set number of iterations (30 used in paper): "))

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
    print("Dataset loaded.")

else:
    print("Not a valid dataset selection. ")
    import sys
    sys.exit()
print("Dataset with " + str(data_class.shape[0]) + " samples loaded.")


def min_max(probabilities, min_y=0, max_y=1):
    res = np.array([min(1.0, max(0.0, item)) for item in probabilities])
    return res

# Number of bins for ECE and MCE.
k = 100
# Run bootstrap-loop to get more stable metrics
# Change sample size and/or sampling rate
# Positive rate? X out of a thousand samples are positive.
# X out of 1024 samples are positive
# positive_rates = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
#                   1024 - 256, 1024 - 128, 1024 - 64, 1024 - 32, 1024 - 16, 1024 - 8, 1024 - 4, 1024 - 2, 1024 - 1]
# x% of samples are positive.
positive_rates = [.1, .2, .3, .4, .5, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 96, 97, 98, 99, 99.5, 99.6, 99.7, 99.8, 99.9]
# Not all sample sizes make sense with the above. Maybe add test requiring that positive_rate / sample_size is
# larger than 1.
sample_sizes = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]  # Maybe skip 100k? Or maybe ENIR benefits from this?
# positive_rates = [10, 50]
# sample_sizes = [1000, 10000]
# The sample sizes and positive rates make a grid search of 29 * 9
# Some of the tests won't be run because the requirement is of at least one negative and one positive sample.
# In addition, some of the experiments will be run on very small sample sizes.
# results = {[item = {} for item in sample_sizes}
results = {}
for sample_size in sample_sizes:
    results[sample_size] = {}
    for positive_rate in positive_rates:
        print("Training for sample size {} and positive rate {}.".format(sample_size, positive_rate))
        results[sample_size][positive_rate] = {}
        # Reset metrics:
        ir_metrics = []
        platt_metrics = []
        bbq_metrics = []
        enir_metrics = []
        beir_metrics = []
        for i in range(n_iterations):
            # This should be the innermost loop to make averaging over iterations easier.
            # Check that sample_size * positive_rate results in at least one negative and one positive sample.
            positive_sample_n = int(positive_rate / 100 * sample_size)
            negative_sample_n = int(sample_size * (1 - positive_rate / 100))
            if positive_sample_n < 1.0 or negative_sample_n < 1.0:
                # Skip entire iteration.
                break
            if sample_size > data_scores.shape[0]:
                # Not enough samples to run test
                break
            # Randomize data
            n_rows = data_scores.shape[0]
            idx = np.random.permutation(range(n_rows))
            data_class = data_class[idx]
            data_scores = data_scores[idx]
            # Aim for 100k samples in testing set (if possible)
            split_index = min(100000, n_rows - sample_size)
            test_class = data_class[:split_index]
            test_scores = data_scores[:split_index]
            training_class = data_class[split_index:]
            training_scores = data_scores[split_index:]
            # Separate positive and negative samples in training set.
            # Take first n out of all positives and negatives. They were shuffled earlier.
            positive_training_class = training_class[training_class][:positive_sample_n]
            negative_training_class = training_class[~training_class][:negative_sample_n]
            positive_training_scores = training_scores[training_class][:positive_sample_n]
            negative_training_scores = training_scores[~training_class][:negative_sample_n]
            # Generate training data of size sample_size and with required positive_rate
            tmp_training_class = np.concatenate((positive_training_class, negative_training_class))
            tmp_training_scores = np.concatenate((positive_training_scores, negative_training_scores))
            # Train all models, and estimate and store metrics on testing set performance

            # IR-model:
            try:
                ir_model = my_ir.train_ir(training_class, training_scores)
                ir_prob = my_ir.predict_ir(ir_model, test_scores)
                ir_metrics.append(isotonic.get_metrics(test_class, ir_prob, k=k))
                # Store also testing set size in addition to metrics.
            except:
                pass

            # Platt-scaling (logistic regression)
            # sklearn uses L2-penalty?!?!?!?!?! WHAAAAT?!?
            try:
                platt_model = logistic_regression.train_logistic_regression(training_class, training_scores)
                platt_prob = logistic_regression.predict_logistic_regression(platt_model, test_scores)
                platt_metrics.append(isotonic.get_metrics(test_class, platt_prob, k=k))
            except:
                pass

            # BBQ
            try:
                octave.push('training_class', training_class, verbose=False)
                octave.push('training_scores', training_scores, verbose=False)
                octave.eval('options.N0 = 2', verbose=False)
                octave.eval("bbq_model = build(training_scores', training_class', options)", verbose=False)
                octave.push('test_scores', test_scores)
                octave.eval("test_prob = predict(bbq_model, test_scores, 1)", verbose=False)
                bbq_prob = octave.pull('test_prob', verbose=False)
                bbq_prob = np.array([item[0] for item in bbq_prob])
                bbq_metrics.append(isotonic.get_metrics(test_class, bbq_prob, k=k))
            except:
                pass

            # ENIR
            # This might crash if there are not enough samples!!!
            try:
                enir_model = enir.enir_build(robjects.FloatVector(training_scores.tolist()), robjects.BoolVector(training_class.tolist()))
                enir_prob = enir.enir_predict(enir_model, robjects.FloatVector(test_scores.tolist()))
                # Convert to numpy.array:
                enir_prob = np.array(enir_prob)
                # Using min_max to deal with "probabilities" that fall outside of [0, 1]
                enir_prob = min_max(enir_prob)
                enir_metrics.append(isotonic.get_metrics(test_class, enir_prob, k=k))
            except:
                pass

            # BEIR
            # Perhaps test BEIR with BIC-score model averaging or simply uniform probability for
            # all models instead of log-likelihood averaging.
            try:
                beir_model = my_ir.train_beir(training_class, training_scores)
                beir_prob = my_ir.predict_beir(beir_model, test_scores)  # There was an error here resulting in AUC-ROC 1.0
                # Sometimes probabilities might be just a tad above 1 due to numeric instability.
                beir_prob = min_max(beir_prob)
                beir_metrics.append(isotonic.get_metrics(test_class, beir_prob, k=k))
            except:
                pass
        # Average results
        tmp_results = {'training_samples': positive_sample_n + negative_sample_n}
        try:
            tmp_results['ir'] = isotonic.average_metrics(ir_metrics)
        except:
            pass
        try:
            tmp_results['platt'] = isotonic.average_metrics(platt_metrics)
        except:
            pass
        try:
            tmp_results['bbq'] = isotonic.average_metrics(bbq_metrics)
        except:
            pass
        try:
            tmp_results['enir'] = isotonic.average_metrics(enir_metrics)
        except:
            pass
        try:
            tmp_results['beir'] = isotonic.average_metrics(beir_metrics)
        except:
            pass
        results[sample_size][positive_rate] = tmp_results
        # results[sample_size][positive_rate].append(tmp_results)

# The loop above goes through all positive rate and sample size combinations specified.
# Printing methods are required...
# Loop through positive rates and sample sizes and match for corresponding fields in results (dict?)


def print_results(results, sample_sizes, positive_rates, method='ir', metric='mse'):
    # return([mse, auc_roc, ece, mce, max_p])
    print('\t\t', end='')
    for positive_rate in positive_rates:
        print(positive_rate, end='\t')
    print('\n', end='')
    for sample_size in sample_sizes:
        print(str(sample_size) + '\t', end='')
        for positive_rate in positive_rates:
            print(results[sample_size][positive_rate][method][metric], end='\t')
            # print(results[sample_size][positive_rate][method][1], end='\t')
        print('\n', end='')


# Perhaps print results into file, or create heat map.
# Heatmap should be colored by entire range of all methods
# Perhaps create separate heatmap where "best method" is color coded for each combination.

def save_heatmap(results, sample_sizes, positive_rates, method='ir', metric='mse', file_name='ir_mse_heatmap.png'):
    # First make the results in question into a numpy array:
    res = []
    methods = ['platt', 'ir', 'bbq', 'enir', 'beir']
    min_val = np.inf
    for sample_size in sample_sizes:
        for positive_rate in positive_rates:
            for method_tmp in methods:
                min_val = min(min_val, results[sample_size][positive_rate][method_tmp][metric])
    max_val = 0
    for sample_size in sample_sizes:
        for positive_rate in positive_rates:
            for method_tmp in methods:
                max_val = max(max_val, results[sample_size][positive_rate][method_tmp][metric])

    for sample_size in sample_sizes:
        res.append([])
        for positive_rate in positive_rates:
            res[-1].append(results[sample_size][positive_rate][method][metric])
    res = np.array(res)
    plt.imshow(res, cmap='hot', vmin=min_val, vmax=max_val)
    plt.ylabel('Sample size')
    plt.xlabel('Positive rate [%]')
    plt.title(method + ' ' + metric)
    plt.xticks(range(len(positive_rates)), tuple(positive_rates))
    plt.yticks(range(len(sample_sizes)), tuple(sample_sizes))
    plt.savefig(file_name)
    # Clear:
    plt.gcf().clear()
    return(True)


def generate_all_heatmaps(results, sample_sizes, positive_rates):
    methods = ['platt', 'ir', 'bbq', 'enir', 'beir']
    metrics = ['mse', 'auc_roc', 'ece', 'mce', 'max_p']
    for method in methods:
        for metric in metrics:
            save_heatmap(results, sample_sizes, positive_rates, method,
                         metric, file_name=method + '_' + metric + '_heatmap.png')
    return(True)


def best_of_all_colormap(results, sample_sizes, positive_rates):
    # I want to print a color to a square in a heatmap where the color corresponds to the
    # method that performed best on a given metric for a positive rate and sample size combination.
    # 1. Generate matrices for all methods for some metric? No.
    # 1. Just find the best method for given metric and positive rate - sample size combination.
    methods = ['platt', 'ir', 'bbq', 'enir', 'beir']
    method_index = {method: index for index, method in enumerate(methods)}
    method_index['None'] = 5  # Add one item for 'none' if there are no results!
    metrics = ['mse', 'auc_roc', 'ece', 'mce', 'max_p']
    best_criteria = {'mse': 'low', 'auc_roc': 'high', 'ece': 'low', 'mce': 'low', 'max_p': 'high'}
    for metric in metrics:
        res_matrix = []  # {metric: [] for metric in metrics}
        for sample_size in sample_sizes:
            res_matrix.append([])
            for positive_rate in positive_rates:
                # Pick the best metric of the given triplet.
                # Find method with min value.
                min_method = 'None'
                min_value = np.inf
                for method in methods:
                    if results[sample_size][positive_rate][method][metric] < min_value:
                        min_value = results[sample_size][positive_rate][method][metric]
                        min_method = method
                max_method = 'None'
                max_value = -np.inf
                for method in methods:
                    if results[sample_size][positive_rate][method][metric] > max_value:
                        max_value = results[sample_size][positive_rate][method][metric]
                        max_method = method
                if best_criteria[metric] == 'low':
                    res_matrix[-1].append(method_index[min_method])
                else:  # best_criteria[metric] == 'high'
                    res_matrix[-1].append(method_index[max_method])
                    # Find method with max value.
        # Plot colormap. Remember labels
        plt.imshow(res_matrix, vmin=0, vmax=5)  # , cmap='hot', vmin=0, vmax=5)
        plt.ylabel('Sample size')
        plt.xlabel('Positive rate [%]')
        plt.title(metric)  # + ', min_value ' + str(min_value) + ', max_value ' + str(max_value))
        plt.xticks(range(len(positive_rates)), tuple(positive_rates))
        plt.yticks(range(len(sample_sizes)), tuple(sample_sizes))
        # How do we add labels so that we know which color is what method?
        plt.savefig(metric + '_' + 'colormap.png')
        plt.gcf().clear()
        # Reset res_matrix. Iterate over all metrics.
    # Next, print a mapping between color and method:
    plt.imshow([[method_index[item]] for item in methods + ['None']], vmin=0, vmax=5, origin='lower')
    ax = plt.gca()
    ax.yaxis.set_ticklabels(['empty'] + [method for method in methods] + ['None'])
    plt.savefig('colorlabels_colormap.png')
    plt.gcf().clear()
    return(True)




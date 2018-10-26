"""
This is a version of Near-Isotonic Regression where we merge bins that are in violation
with the following bin following a path that is defined by number of samples in the bin.
Whenever a bin is in violation wiht the next one and has the smallest number of samples
of all violating bins, this will be merged with the next one.

Notes:
-We could potentially deal with the violations at both ends of the bin range.
-Maybe from there, we can get to L-0 regularization.
-Maybe merge bins that are neighboring that map to the same probability.
"""

import numpy as np
from scipy.interpolate import interp1d


def train_enir_n(data_class, data_scores, no_gaps=False, smoothing=0):
    # Function for training enir model.
    # Set 'smoothing' to 1 for Laplace-smoothing or 1/2 for Krichesky-Trofimov
    # Perhaps opt for an approach where we allow for a violation only if there are at least
    # M samples in a violating bin.
    # 1. Sort samples:
    print("Sorting.")
    data_idx = np.argsort(data_scores)
    data_scores = data_scores[data_idx]
    data_class = data_class[data_idx]
    # 2. Bin samples:
    print("Binning.")
    probabilities = [{'k': int(data_class[0]), 'n': 1, 'p': float(data_class[0]),
                      'score_min': data_scores[0], 'score_max': data_scores[0]}]
    for item_score, item_class in zip(data_scores[1:], data_class[1:]):
        if item_score == probabilities[-1]['score_min']:
            # Add item to last bin.
            probabilities[-1]['k'] = probabilities[-1]['k'] + int(item_class)
            probabilities[-1]['n'] = probabilities[-1]['n'] + 1
            probabilities[-1]['score_max'] = item_score  # This should also remain constant.
            # score_min remains constant for bin.
            probabilities[-1]['p'] = float(probabilities[-1]['k']) / float(probabilities[-1]['n'])
        else:
            # Else, create new bin.
            probabilities.append({'k': int(item_class), 'n': 1, 'p': float(item_class),
                                  'score_min': item_score, 'score_max': item_score})
    print("Binning done.")
    if no_gaps:
        # Remove gaps from model
        for i in range(len(probabilities) - 1):
            tmp_score = probabilities[i]['max_score'] + probabilities[i + 1]['min_score']
            probabilities[i]['max_score'] = tmp_score
            probabilities[i + 1]['min_score'] = tmp_score

    def get_violations(probabilities):
        violations = [i for i in range(len(probabilities) - 1) if (probabilities[i]['p'] - probabilities[i + 1]['p']) > 0]
        return(np.array(violations))

    def merge_bins(probabilities, bins_to_merge):
        new_probabilities = []
        new_probabilities.append(probabilities[0])
        for i in range(1, len(probabilities)):
            if i in (np.array(bins_to_merge) + 1):  # If previous bin needs to be merged with next.
                new_probabilities[-1]['k'] = new_probabilities[-1]['k'] + probabilities[i]['k']
                new_probabilities[-1]['n'] = new_probabilities[-1]['n'] + probabilities[i]['n']
                new_probabilities[-1]['score_max'] = probabilities[i]['score_max']
                # score_min remains the same
                new_probabilities[-1]['p'] = new_probabilities[-1]['k'] / float(new_probabilities[-1]['n'])
            else:
                new_probabilities.append(probabilities[i])
        return(new_probabilities)

    def create_model(probabilities, smoothing=smoothing, y_min=0, y_max=1):
        # Create interpolation model for one M.
        # NOTE: We might want to change y_min and y_max to e.g. 1e-3 and 1-1e-3.
        y = []
        x = []
        for item in probabilities:
            y.append((item['k'] + smoothing) / float(item['n'] + 2 * smoothing))
            y.append((item['k'] + smoothing) / float(item['n'] + 2 * smoothing))
            x.append(item['score_min'])
            x.append(item['score_max'])
        model_tmp = model_tmp = interp1d(x=x, y=y, bounds_error=False, assume_sorted=True, fill_value=(y_min, y_max))
        return(model_tmp)

    def get_bic_score(probabilities, smoothing=smoothing, y_min=1e-3, y_max=1 - 1e-3):
        # Estimate BIC score
        log_likelihood = 0
        for item in probabilities:
            p_tmp = min(y_max, max(y_min, item['k'] / float(item['n'])))
            log_likelihood += item['k'] * np.log(p_tmp) + (item['n'] - item['k']) * np.log(1 - p_tmp)
        # Use log_likelihood and other parameters to estimate bic_score:
        # Not sure where the log(2*pi) comes from. Copied from Naeini's model.
        bic_score = len(probabilities) * np.log(sum([item['n'] for item in probabilities])) + np.log(2 * np.pi) - 2 * log_likelihood
        return(bic_score)

    def elbow(bayes_factors, alpha=0.005):
        # As implemented by Naeini & al.
        # Function for selecting subset of models.
        relative_log_likelihood = np.exp(bayes_factors / -2)
        var_relative_log_likelihood = np.var(relative_log_likelihood)  # variance of relative log likelihood
        # 1. Order scores
        order_idx = np.argsort(relative_log_likelihood)[::-1]  # Order is descending now!
        # 2. Create index that catches where the change between two consecutive ordered scores
        include_idx = [0]  # Include at least one model, the one with best score
        for i in range(1, len(relative_log_likelihood)):
            if (relative_log_likelihood[order_idx[i - 1]] - relative_log_likelihood[order_idx[i]]) / var_relative_log_likelihood > alpha:
                include_idx.append(i)
            else:
                break
        include_idx = np.array(include_idx)
        model_idx = order_idx[include_idx]
        return(model_idx)

    def get_merges(probabilities, violations):
        # If a bin can be combined with both neighboring bins, the one that results in a smaller
        # bin (by #samples) should be prefered.
        # merge score is [min(samples(bin(i)), samples(bin(i+1))), samples(bin(i, i+1))]
        # If the first is equal, the second decides which one is implemented.
        merge_scores = [{'min_n': min(probabilities[violations[0]]['n'], probabilities[violations[0] + 1]['n']),
                         'new_n': (probabilities[violations[0]]['n'] + probabilities[violations[0] + 1]['n']),
                         'merge_index': violations[0]}]
        for idx in violations:
            merge_scores.append({'min_n': min(probabilities[idx]['n'], probabilities[idx + 1]['n']),
                                 'new_n': (probabilities[idx]['n'] + probabilities[idx + 1]['n']),
                                 'merge_index': idx})
        min_n = min([item['min_n'] for item in merge_scores])
        # Pick out all items that have minimum n. Then find the ones that have minimum new_n.
        merge_index_tmp = [item for item in merge_scores if item['min_n'] == min_n]
        min_new_n = min([item['new_n'] for item in merge_index_tmp])
        merge_index = [item['merge_index'] for item in merge_index_tmp if item['new_n'] == min_new_n]
        return(merge_index)

    violations = get_violations(probabilities)
    model_ensemble = []
    while(len(violations) > 0):
        # print(len(violations))
        # Find violating bin with smallest number of samples
        # min_n = min([item['n'] for item, i in zip(probabilities, range(len(probabilities))) if i in violations])
        # next_merges = [i for i in violations if probabilities[i]['n'] == min_n]
        next_merges = get_merges(probabilities, violations)
        # Merge violating bins. Also merge bins that map to same probabilities.
        probabilities = merge_bins(probabilities, next_merges)
        equal_probability_bins = [i for i in range(len(probabilities) - 1) if probabilities[i]['p'] == probabilities[i + 1]['p']]
        if len(equal_probability_bins) > 0:
            probabilities = merge_bins(probabilities, equal_probability_bins)
        # Store model (perhaps estimate BIC, maybe add option for smoothing)
        tmp_model = create_model(probabilities, smoothing=smoothing)
        tmp_model_bic_score = get_bic_score(probabilities, smoothing=smoothing)
        model_ensemble.append({'model': tmp_model, 'bic_score': tmp_model_bic_score})
        # Find new violations
        violations = get_violations(probabilities)
    # Estimate Bayes factors of models.
    min_bic_score = min([item['bic_score'] for item in model_ensemble])
    for i in range(len(model_ensemble)):
        model_ensemble[i]
    for item in model_ensemble:
        item['bayes_factor'] = item['bic_score'] - min_bic_score
    # Pruning
    ensemble_idx = elbow(np.array([item['bayes_factor'] for item in model_ensemble]))
    model_ensemble = [item for i, item in zip(range(len(model_ensemble)), model_ensemble) if i in ensemble_idx]
    return(model_ensemble)


def predict_enir_n(models, data_scores, model_averaging=True, model_idx=-1):
    # Start by predicting using only the last model. The model is an interpolation model.
    if model_averaging:
        # DUE TO NUMERIC INSTABILITY, THE TOTAL WEIGHT MIGHT IN RARE CASES EXCEED 1 SLIGHTLY.
        # The '-2' returns the factor to the format proposed by Schwarz (1978).
        normalizing_factor = sum([np.exp(item['bayes_factor'] / -2) for item in models])
        probabilities = np.array([item['model'](data_scores) * np.exp(item['bayes_factor'] / -2) / normalizing_factor for item in models]).sum(0)
    else:
        probabilities = models[model_idx]['model'](data_scores)  # Last model is full IR fit.
    return(probabilities)

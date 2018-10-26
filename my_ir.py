"""
This will become a version of IR that does _not_ have the bug that exists in
sklearn.isotonic.IsotonicRegression.
This version should also be fast. It is based on the observation that under
no circumstances will the borders between IR bins violate, i.e. any violating
bins can be merged without caring about order.

Notes:
-I think the problem in sklearn comes from not noticing that samples with
equal scores _must_ map to the same probability in IR.
-This version is correct (AFAIK...). It matches the IR model created by
my_enir and results in better training set performance than
sklearn.isotonic.IsotonicRegression.
-There is probably some optimization to be done, although this version
runs on dataset 1 (133k training samples) in 8 seconds.
-The version 2 below runs in around one second.
"""

import numpy as np
from scipy.interpolate import interp1d


def train_ir(data_class, data_scores, no_gaps=False, smoothing=0, include_no_gaps_model=True):
    # This version of Isotonic Regression is quick and runs in
    # O(n*log(n)) for the sorting algorithm and O(n) for PAVA.
    # The biggest slowing factor is probably the removal of elements
    # from a list using del.
    data_idx = np.argsort(data_scores)
    data_scores = data_scores[data_idx]
    data_class = data_class[data_idx]
    # 2. Bin samples:
    bin_table = [{'k': int(data_class[0]), 'n': 1, 'p': float(data_class[0]),
                  'score_min': data_scores[0], 'score_max': data_scores[0]}]
    for item_score, item_class in zip(data_scores[1:], data_class[1:]):
        if item_score == bin_table[-1]['score_min']:
            # Add item to last bin.
            bin_table[-1]['k'] = bin_table[-1]['k'] + int(item_class)
            bin_table[-1]['n'] = bin_table[-1]['n'] + 1
            bin_table[-1]['score_max'] = item_score  # This should also remain constant.
            # score_min remains constant for bin.
            bin_table[-1]['p'] = float(bin_table[-1]['k']) / float(bin_table[-1]['n'])
        else:
            # Else, create new bin.
            bin_table.append({'k': int(item_class), 'n': 1, 'p': float(item_class),
                              'score_min': item_score, 'score_max': item_score})
    # Next go through the bin table and resolve conflicts one by one until none are left.
    i = 0
    while i < len(bin_table) - 1:
        if bin_table[i]['p'] >= bin_table[i + 1]['p']:  # If equal, merge bins.
            # A violation or a spot when two bins can be treated as the same.
            # The equality sign might create a smaller interpolation model.
            bin_table[i]['k'] += bin_table[i + 1]['k']
            bin_table[i]['n'] += bin_table[i + 1]['n']
            bin_table[i]['score_max'] = bin_table[i + 1]['score_max']
            bin_table[i]['p'] = bin_table[i]['k'] / float(bin_table[i]['n'])
            # Drop bin i + 1:
            del bin_table[i + 1]
            # Check that previous bin is not in violation with this new bin.
            if i > 0:
                i -= 1
        else:
            i += 1

    def create_model(bin_table, smoothing=smoothing, y_min=0, y_max=1):
        # Create interpolation model for one M.
        # NOTE: We might want to change y_min and y_max to e.g. 1e-3 and 1-1e-3.
        y = []
        x = []
        for item in bin_table:
            y.append((item['k'] + smoothing) / float(item['n'] + 2 * smoothing))
            y.append((item['k'] + smoothing) / float(item['n'] + 2 * smoothing))
            x.append(item['score_min'])
            x.append(item['score_max'])
        model_tmp = model_tmp = interp1d(x=x, y=y, bounds_error=False, assume_sorted=True, fill_value=(y_min, y_max))
        return(model_tmp)
    # Format result into an interpolation model:
    model = create_model(bin_table, smoothing)

    if include_no_gaps_model:
        # Remove gaps from model
        for i in range(len(bin_table) - 1):
            tmp_score = (bin_table[i]['score_max'] + bin_table[i + 1]['score_min']) / 2
            bin_table[i]['score_max'] = tmp_score
            bin_table[i + 1]['score_min'] = tmp_score
        no_gaps_model = create_model(bin_table, smoothing=smoothing)

    if include_no_gaps_model:
        return({'model': model, 'no_gaps_model': no_gaps_model})
    else:
        return({'model': model})


def predict_ir(model, data_scores, use_no_gaps_model=False):
    # Start by predicting using only the last model. The model is an interpolation model.
    if use_no_gaps_model:  # Use model with no gaps.
        probabilities = model['no_gaps_model'](data_scores)
    else:
        probabilities = model['model'](data_scores)
    return(probabilities)


def train_beir(data_class, data_scores, n_models=100, sampling_rate=.95, y_min=1e-3, y_max=1 - 1e-3):
    # Create a bootstrap ensemble of IR models with model averaging.
    # Using log-likelihood for model averaging.
    model = []
    n_samples = len(data_class)
    for i in range(n_models):  # THIS COULD ALSO BE DONE IN PARALLEL WITH pool() or similar.
        # Perhaps say something about progress.
        # Resample data
        # print("Working on {0} of {1} models.".format(i, n_models))
        idx = np.random.randint(low=0, high=n_samples, size=int(sampling_rate * n_samples))
        tmp_data_class = data_class[idx]
        tmp_data_scores = data_scores[idx]
        # Build ir-model
        tmp_model = train_ir(tmp_data_class, tmp_data_scores, include_no_gaps_model=True)
        # BIC-score and log-likelihood must be estimated here as we want to do it on the entire training set.
        # Does bic-score make any sense, as we are estimating log-likelihood on a dataset that was partly
        # unavailable during training?
        data_prob = predict_ir(tmp_model, data_scores, use_no_gaps_model=True)
        data_prob_idx = np.argsort(data_prob)
        data_prob = data_prob[data_prob_idx]
        data_class = data_class[data_prob_idx]
        table = [{'k': int(data_class[0]), 'n': 1, 'p': float(data_class[0])}]
        for item_prob, item_class in zip(data_prob[1:], data_class[1:]):
            if item_prob == table[-1]['p']:
                # Add item to last 'bin'
                table[-1]['k'] = table[-1]['k'] + int(item_class)
                table[-1]['n'] = table[-1]['n'] + 1
                table[-1]['p'] = table[-1]['k'] / float(table[-1]['n'])
            else:
                # Create new element
                table.append({'k': int(item_class), 'n': 1, 'p': float(item_class)})
        log_likelihood = 0
        for item in table:
            log_likelihood += item['k'] * np.log(min(y_max, max(y_min, item['p'])))
            log_likelihood += (item['n'] - item['k']) * np.log(min(y_max, max(y_min, 1 - item['p'])))
        # Append
        model.append({'model': tmp_model, 'log_likelihood': log_likelihood})
    # Add relative log_likelihood
    max_log_likelihood = max([item['log_likelihood'] for item in model])
    for item in model:
        item['relative_log_likelihood'] = item['log_likelihood'] - max_log_likelihood
    # Return collection of models.
    return(model)


def predict_beir(model, data_scores, model_averaging='uniform'):
    if model_averaging == 'uniform':
        n_models = len(model)
        probabilities = np.array([item['model']['model'](data_scores) / n_models for item in model]).sum(0)
    elif model_averaging == 'log_likelihood':
        normalizing_factor = sum([np.exp(item['relative_log_likelihood']) for item in model])
        probabilities = np.array([item['model']['model'](data_scores) * np.exp(item['relative_log_likelihood']) / normalizing_factor for item in model]).sum(0)
    return(probabilities)


# Slow version of IR here. Done faster in train_ir() above.
# def train_ir_slow(data_class, data_scores, no_gaps=False, smoothing=0, include_no_gaps_model=False):
#     # Function for training enir model.
#     # Set 'smoothing' to 1 for Laplace-smoothing or 1/2 for Krichesky-Trofimov
#     # Perhaps opt for an approach where we allow for a violation only if there are at least
#     # M samples in a violating bin.
#     # 1. Sort samples:
#     data_idx = np.argsort(data_scores)
#     data_scores = data_scores[data_idx]
#     data_class = data_class[data_idx]
#     # 2. Bin samples:
#     probabilities = [{'k': int(data_class[0]), 'n': 1, 'p': float(data_class[0]),
#                       'score_min': data_scores[0], 'score_max': data_scores[0]}]
#     for item_score, item_class in zip(data_scores[1:], data_class[1:]):
#         if item_score == probabilities[-1]['score_min']:
#             # Add item to last bin.
#             probabilities[-1]['k'] = probabilities[-1]['k'] + int(item_class)
#             probabilities[-1]['n'] = probabilities[-1]['n'] + 1
#             probabilities[-1]['score_max'] = item_score  # This should also remain constant.
#             # score_min remains constant for bin.
#             probabilities[-1]['p'] = float(probabilities[-1]['k']) / float(probabilities[-1]['n'])
#         else:
#             # Else, create new bin.
#             probabilities.append({'k': int(item_class), 'n': 1, 'p': float(item_class),
#                                   'score_min': item_score, 'score_max': item_score})
#     if no_gaps:
#         # Remove gaps from model
#         for i in range(len(probabilities) - 1):
#             tmp_score = (probabilities[i]['score_max'] + probabilities[i + 1]['score_min']) / 2
#             probabilities[i]['score_max'] = tmp_score
#             probabilities[i + 1]['score_min'] = tmp_score

#     def get_violations(probabilities):
#         violations = [i for i in range(len(probabilities) - 1) if (probabilities[i]['p'] - probabilities[i + 1]['p']) > 0]
#         return(np.array(violations))

#     def merge_bins(probabilities, bins_to_merge):
#         new_probabilities = []
#         new_probabilities.append(probabilities[0])
#         for i in range(1, len(probabilities)):
#             if i in (np.array(bins_to_merge) + 1):  # If previous bin needs to be merged with next.
#                 new_probabilities[-1]['k'] = new_probabilities[-1]['k'] + probabilities[i]['k']
#                 new_probabilities[-1]['n'] = new_probabilities[-1]['n'] + probabilities[i]['n']
#                 new_probabilities[-1]['score_max'] = probabilities[i]['score_max']
#                 # score_min remains the same
#                 new_probabilities[-1]['p'] = new_probabilities[-1]['k'] / float(new_probabilities[-1]['n'])
#             else:
#                 new_probabilities.append(probabilities[i])
#         return(new_probabilities)

#     def create_model(probabilities, smoothing=smoothing, y_min=0, y_max=1):
#         # Create interpolation model for one M.
#         # NOTE: We might want to change y_min and y_max to e.g. 1e-3 and 1-1e-3.
#         y = []
#         x = []
#         for item in probabilities:
#             y.append((item['k'] + smoothing) / float(item['n'] + 2 * smoothing))
#             y.append((item['k'] + smoothing) / float(item['n'] + 2 * smoothing))
#             x.append(item['score_min'])
#             x.append(item['score_max'])
#         model_tmp = model_tmp = interp1d(x=x, y=y, bounds_error=False, assume_sorted=True, fill_value=(y_min, y_max))
#         return(model_tmp)

#     # Find violating bins:
#     violations = get_violations(probabilities)
#     while(len(violations) > 0):
#         # Merge violating bins. Also merge bins that map to same probabilities.
#         probabilities = merge_bins(probabilities, violations)
#         # Find new violations
#         violations = get_violations(probabilities)

#     # For bic_score, we need to merge bins that have equal probabilities.
#     equal_probability_bins = [i for i in range(len(probabilities) - 1) if probabilities[i]['p'] == probabilities[i + 1]['p']]
#     if len(equal_probability_bins) > 0:
#         probabilities = merge_bins(probabilities, equal_probability_bins)

#     # Create model.
#     model = create_model(probabilities, smoothing=smoothing)

#     if include_no_gaps_model:
#         # Make a no-gaps -model.
#         for i in range(len(probabilities) - 1):
#             tmp_score = (probabilities[i]['score_max'] + probabilities[i + 1]['score_min']) / 2
#             probabilities[i]['score_max'] = tmp_score
#             probabilities[i + 1]['score_min'] = tmp_score
#         no_gaps_model = create_model(probabilities, smoothing=smoothing)
#         return({'model': model, 'no_gaps_model': no_gaps_model})
#     else:
#         return({'model': model})

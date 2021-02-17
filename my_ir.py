"""
This is a version of IR that does _not_ have the bug that exists in
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
runs on dataset 1 (133k training samples) in a second.
"""

import numpy as np
from scipy.interpolate import interp1d


def train_ir(data_class, data_scores, no_gaps=False, smoothing=0, include_no_gaps_model=True):
    """
    A version of isotonic regression that is reasonably quick and runs in O(n*log(n))
    for the sorting algorithm and O(n) for PAVA.
    The biggest slowing factor is probably the removal of elements
    from a list using del.

    Args:
    data_class (np.array([])): Array of class-labels for samples.
    data_scores (np.array([])): Array of floats for samples.
    smoothing (float): Set to 1 for Lambda-smoothing, 1/2 for Krichesky-Trofimov.
    no_gaps (bool): With no_gaps set to False, this will produce an isotonic
     regression model that interpolates between bins. With True, the IR model
     will be modified so that there are no gaps between bins.
    include_no_gaps_model (bool): With this set to True, the function will return
     one IR model with gaps and one without.
    """
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
    """
    This function predicts probabilities from scores using an IR model.

    Args:
    model: A model as returned by train_ir().
    data_scores (np.array([])): An array of scores for samples to be
     turned into probabilities using the model.
    use_no_gaps_model (bool): If the model was trained with
     include_no_gaps_model=True, then this flag can be set to select which
     model to use for predictions.
    """
    # Start by predicting using only the last model. The model is an interpolation model.
    if use_no_gaps_model:  # Use model with no gaps.
        probabilities = model['no_gaps_model'](data_scores)
    else:
        probabilities = model['model'](data_scores)
    return(probabilities)


def train_beir(data_class, data_scores, n_models=100, sampling_rate=.95, y_min=1e-3, y_max=1 - 1e-3):
    """
    Function for creating a bootstrap ensemble of IR models with model averaging using
    log-likelihood.

    Args:
    data_class (np.array([bool])): Array of class-labels. True indicates positive.
    data_scores (np.array([float])): Array of scores for samples.
    n_models (int): Number of models in ensemble.
    sampling_rate (float): Fraction of samples to use in one bootstrap iteration. Value
     in ]0, 1].
    y_min (float): Smallest value to be predicted by model (capped at this value). 
     In [0, 1[
    y_max (float): Largest value to predicted by model. In ]y_min, 1].
    """
    model = []
    n_samples = len(data_class)
    for i in range(n_models):  # THIS COULD ALSO BE DONE IN PARALLEL WITH pool() or similar.
        # Perhaps say something about progress.
        # Resample data
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
    """
    Function for creating predictions from scores using a BEIR model.

    Args:
    model: Model as returned by train_beir().
    data_scores (np.array([float])): Array of scores for samples to turn
     into probabilities.
    model_averaging {'uniform', 'log_likelihood'}: Flag to decide whether
     the model averaging should be done using log-likelihood or uniform
     weights for models.
    """
    if model_averaging == 'uniform':
        n_models = len(model)
        probabilities = np.array([item['model']['model'](data_scores) / n_models for item in model]).sum(0)
    elif model_averaging == 'log_likelihood':
        normalizing_factor = sum([np.exp(item['relative_log_likelihood']) for item in model])
        probabilities = np.array([item['model']['model'](data_scores) * np.exp(item['relative_log_likelihood']) / normalizing_factor for item in model]).sum(0)
    return(probabilities)

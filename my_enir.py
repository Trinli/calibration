"""
Ensemble of near-isotonic regression (ENIR) for calibration
The purpose of this code is to study ENIR. There is some discrepancy
in the Matlab-code released by Naeini & al. where some samples
sometimes get mapped to values outside of [0, 1] - which obsviously
is not wanted as the outputs should correspond to probabilities.
This code might also serve as a Python implementation of ENIR for
production purposes.

Notes:
-Perhaps add Laplace-smoothing
-Alternatively try +1/2 smoothing
"""

import numpy as np
from scipy.interpolate import interp1d


def train_enir(data_class, data_scores, y_min=0.0, y_max=1.0, no_gaps=False, laplace_smoothing=False, max_likelihood=False, pruning=True):
    # 'laplace_smoothing' causes bins to have frequencies that are smoothed, but not affected by the lambda.
    # 'max_likelihood' follows the same solution path as ENIR but sets the bin probabilities to the
    # relative frequency in the bins (i.e. "L0-regularization").
    # First a bunch of auxiliary functions:

    def update_probabilities(probabilities, slopes, delta_lambda):
        # Version in pseudo-code in paper.
        new_probabilities = []
        for item, slope in zip(probabilities, slopes):
            new_probabilities.append(item)
            tmp = new_probabilities[-1]['p'] + slope * delta_lambda
            if tmp > 1:
                print("-" * 40)
                print("Anomaly with probability while updating probabilities.")
                print(new_probabilities[-1]['p'])
                print(type(slope))
                print(slope)
                print(delta_lambda)
            new_probabilities[-1]['p'] = tmp
        return(new_probabilities)

    def merge_bins(probabilities, bins_to_merge=[]):
        # Function for merging bins in list.
        new_probabilities = []
        bins_to_merge = np.array(bins_to_merge)
        # Walk through range(len(probabilities)), add elements one by one to new_probabilities,
        # and merge with previous bin as needed.
        for i in range(len(probabilities)):
            if i in (bins_to_merge + 1):
                # The first bin cannot be here!
                if np.abs(probabilities[i]['p'] - probabilities[i - 1]['p']) > .0000001:
                    print("Anomaly in bin probabilities while merging bins.")
                    print(probabilities[i - 1]['p'])
                    print(probabilities[i]['p'])
                # Merge to previous bin.
                new_probabilities[-1]['n'] = new_probabilities[-1]['n'] + probabilities[i]['n']
                new_probabilities[-1]['k'] = new_probabilities[-1]['k'] + probabilities[i]['k']
                new_probabilities[-1]['score_max'] = probabilities[i]['score_max']
                # score_min remains the same, p should be the same in both bins.
            else:
                new_probabilities.append(probabilities[i])
        return(new_probabilities)

    def get_violations(probabilities):
        # Auxiliary function for violations.
        # 'violations' contain info about violations of monotonicity in the binning model. Note
        # that the indexing is one off, i.e. that violations[0] does not correspond to any violation.
        violations = [0]  # First element set to zero to simplify calculations.
        for i in range(len(probabilities) - 1):
            if probabilities[i]['p'] > probabilities[i + 1]['p']:
                violations.append(1)
            else:
                violations.append(0)
        violations.append(0)  # This is needed in update_probabilities().
        return(violations)

    def get_slopes(violations, probabilities):
        # Equation (7) in ENIR paper
        a = []
        for i in range(len(probabilities)):
            # violations[i] below correspond to violations_{i-1} in paper
            a.append((violations[i] - violations[i + 1]) / probabilities[i]['n'])
        return(a)

    def get_lambda_values(probabilities, slopes, lambda_value):
        # Merging values. Equation (8) in ENIR paper.
        l = []
        for i in range(len(probabilities) - 1):  # Last bin will never be merged to a "next" bin.
            delta_slopes = slopes[i + 1] - slopes[i]
            if delta_slopes == 0:  # Handle division by zero
                if probabilities[i]['p'] == probabilities[i + 1]['p']:
                    l.append(lambda_value)
                    # print("Anomaly with lambda values in get_lambda_values()")
                    # If we get here, the related bins should be merged next (this should be ok),
                    # and bin probabilities should remain constant (both had the same probability).
                    # As a consequence lambda_tmp - lambda_min == 0
                else:
                    l.append(np.inf)
            else:
                l.append((probabilities[i]['p'] - probabilities[i + 1]['p']) / delta_slopes + lambda_value)
        return(l)

    def get_bic_score(probabilities, model_probabilities, p_min=1e-3, p_max=1 - 1e-3):
        # Function for estimating the bayesian information criterion.
        # Messy implementation. Clean?
        def get_log_likelihood(probabilities, model_probabilities=model_probabilities, p_min=p_min, p_max=p_max):
            # The likelihood of a bin is p^k*(1-p)^(n-k):
            log_likelihood = 0
            for i in range(len(probabilities)):
                log_likelihood += probabilities[i]['k'] * np.log(max(min(model_probabilities[i], p_max), p_min))
                log_likelihood += (probabilities[i]['n'] - probabilities[i]['k']) * np.log(max(p_min, min(p_max, (1 - model_probabilities[i]))))
            return(log_likelihood)

        log_likelihood = get_log_likelihood(probabilities)
        # Naeini & al. adds np.log(2 * np.pi) to the BIC-score. No idea why. It vanishes when
        # the bayes factor is calculated. Naeini also uses Laplace-smoothing in the bins.
        bic_score = len(probabilities) * np.log(sum([item['n'] for item in probabilities])) - 2 * log_likelihood
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
        model_idx = [int(item) for item in model_idx]  # Remove int64 type...
        return(model_idx)

    # First binning model is one with equal scoring samples
    # in one bin. NIR assumes that samples are equally spaced and suggest dividing by
    # the difference in spacing between two consequtive samples if this is not the case.
    # In ENIR, some samples have the exact same scoring(!). We do not actually care about
    # the space between these, but reduce them to ranks. Consequently, equal scoring samples
    # will have the exact same rank and not putting these in the same bins will result in
    # infinite loss for any non-zero lambda. Hence, we must start from the following:
    # (this in contrast to ENIR-paper where the authors imply that samples are placed
    # one per bin at start).

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
            probabilities[-1]['score_max'] = item_score
            # score_min remains constant for bin.
            probabilities[-1]['p'] = float(probabilities[-1]['k']) / float(probabilities[-1]['n'])
        else:
            # Else, create new bin.
            probabilities.append({'k': int(item_class), 'n': 1, 'p': float(item_class),
                                  'score_min': item_score, 'score_max': item_score})
    print("Binning done.")

    # Set parameters:
    lambda_tmp = 0.0
    models = []
    violations = get_violations(probabilities)
    print("Training my_enir.")
    while(sum(violations) > 0):
        # while(lambda_tmp < 1000 and len(probabilities) > 2):  # This criterion often ends up with very few bins as one merge iteration may cover multiple bins!
        # for j in range(36):
        # Find the bin(s) with the smallest lambda, i.e. the one to be merged next.
        slopes = get_slopes(violations, probabilities)
        lambda_values = get_lambda_values(probabilities, slopes, lambda_tmp)
        # QUICK FIX (SEE PAGE 4 IN NIR-PAPER) (next lambda-value must be larger than previous)
        # The row below must be wrong. It would gauarantee that the if-statement below always
        # evaluates to True.
        # lambda_values = [item if item > lambda_tmp else np.inf for item in lambda_values]
        # Next merging value:
        lambda_min = min(lambda_values)
        if(lambda_min < lambda_tmp):
            print("Anomaly with lambda-values.")
            break
        # Find all matching occurrences in lambda_values:
        bins_to_merge = [i for i, lambda_value in enumerate(lambda_values) if lambda_value == lambda_min]
        # Function update_probabilities() affects the input object... i.e. new_probabilities and
        # probabilities will be identical.
        probabilities = update_probabilities(probabilities, slopes, lambda_min - lambda_tmp)
        # if(max([item['p'] for item in probabilities]) > 1.0):
        #     break
        # else:
        #     probabilities = new_probabilities
        # probabilities = update_probabilities(probabilities, slopes, lambda_tmp - lambda_min)
        # ERROR RIGHT HERE!
        # The merge_bins() -function does not update the probabilities correctly according to the required formulas.
        probabilities = merge_bins(probabilities, bins_to_merge)
        lambda_tmp = lambda_min
        # bic_score = get_bic_score(probabilities)  # Min-max the zeroes and ones out.
        # Format stuff for creating interpolation model:
        model_boundaries = []
        model_probabilities = []
        # THE CODE BELOW REMOVES THE GAPS BETWEEN BINS. THIS IS NOT NECESSARILY DESIRED.
        if no_gaps:
            for i in range(len(probabilities)):
                if i == 0:
                    # model_boundaries.append(-np.inf)  # Should this be zero-indexed?
                    model_boundaries.append(probabilities[0]['score_min'])
                    model_boundaries.append((probabilities[0]['score_max'] + probabilities[1]['score_min']) / 2)
                elif i == len(probabilities) - 1:
                    model_boundaries.append((probabilities[i - 1]['score_max'] + probabilities[i]['score_min']) / 2)
                    # model_boundaries.append(np.inf)
                    model_boundaries.append(probabilities[i]['score_max'])
                else:
                    model_boundaries.append((probabilities[i - 1]['score_min'] + probabilities[i]['score_min']) / 2)
                    model_boundaries.append((probabilities[i]['score_max'] + probabilities[i + 1]['score_min']) / 2)
                if laplace_smoothing:
                    model_probabilities.append((probabilities[i]['k'] + 1) / float(probabilities[i]['n'] + 2))
                    model_probabilities.append((probabilities[i]['k'] + 1) / float(probabilities[i]['n'] + 2))
                elif max_likelihood:
                    model_probabilities.append((probabilities[i]['k']) / float(probabilities[i]['n']))
                    model_probabilities.append((probabilities[i]['k']) / float(probabilities[i]['n']))
                else:
                    model_probabilities.append(probabilities[i]['p'])
                    model_probabilities.append(probabilities[i]['p'])
        else:  # If gaps allowed. Should probably be default.
            for i in range(len(probabilities)):
                model_boundaries.append(probabilities[i]['score_min'])
                model_boundaries.append(probabilities[i]['score_max'])
                if laplace_smoothing:
                    model_probabilities.append((probabilities[i]['k'] + 1) / float(probabilities[i]['n'] + 2))
                    model_probabilities.append((probabilities[i]['k'] + 1) / float(probabilities[i]['n'] + 2))
                elif max_likelihood:
                    model_probabilities.append((probabilities[i]['k']) / float(probabilities[i]['n']))
                    model_probabilities.append((probabilities[i]['k']) / float(probabilities[i]['n']))
                else:
                    model_probabilities.append(probabilities[i]['p'])
                    model_probabilities.append(probabilities[i]['p'])
        # Messy implementation with bic-score! Clean up.
        bic_score = get_bic_score(probabilities, model_probabilities)
        model_tmp = interp1d(x=model_boundaries, y=model_probabilities, bounds_error=False, assume_sorted=True, fill_value=(y_min, y_max))
        # model_tmp._fill_value_below = min(model_probabilities)
        # model_tmp._fill_value_above = max(model_probabilities)
        models.append({'model': model_tmp, 'bic_score': bic_score})
        # print(len(probabilities))
        violations = get_violations(probabilities)
    # Estimate bayes factors instead of BIC scores:
    min_bic_score = min([item['bic_score'] for item in models])
    for i in range(len(models)):
        models[i]['bayes_factor'] = models[i]['bic_score'] - min_bic_score
    if pruning:
        model_idx = elbow(np.array([item['bayes_factor'] for item in models]))
        models = [item for i, item in zip(range(len(models)), models) if i in model_idx]
    return(models)


def predict_enir(models, data_scores, model_averaging=True, model_idx=-1):
    # Start by predicting using only the last model. The model is an interpolation model.
    if not model_averaging:
        probabilities = models[model_idx]['model'](data_scores)  # Last model is full IR fit.
    else:
        # Use model averaging
        # DUE TO NUMERIC INSTABILITY, THE TOTAL WEIGHT MIGHT IN RARE CASES EXCEED 1 SLIGHTLY.
        # The '-2' returns the factor to the format proposed by Schwarz (1978).
        normalizing_factor = sum([np.exp(item['bayes_factor'] / -2) for item in models])
        probabilities = np.array([item['model'](data_scores) * np.exp(item['bayes_factor'] / -2) / normalizing_factor for item in models]).sum(0)
    return(probabilities)

# Test 2 here produces a result that is different from the isotonic regression model produced by
# sklearn.isotonic. The min and max scores (bin boundaries) for the first three bins match in both
# cases. The 'p'-values for this version match the natural frequency of positive samples in the bins,
# hence, the probabilities for the sklearn.isotonic-version must be wrong for bin 2 and 3.
# ?!?

# Calibration

Calibration" is the process of turning e.g. machine learning predictions into probabilities.
This is particularly useful whenever the predictions are used downstream to estimate expected
values and similar. There are a number of calibration methods. This repository contains a handful 
I have decided to implement myself both for learning to understand how and why they actually work.
These are also suitable for model comparisons.

This repository contains contains code and data for a paper to be published at PAKDD (www.pakdd2021.org).
Specifically in isotonic.py there are the train_rcir() and predict_rcir()-functions that
were the main contribution in the paper. load_pickle() can be used to load the datasets that
were used in the experiments that reside in the data-folder.

Models included:

isotonic.py: Reliably calibrated isotonic regression - The model used in the PAKDD 2021-paper. A variant
of isotonic regression that provides guarantees for narrowness of credible intervals of predictions.

my_ir.py: Isotonic regression - It turns out the version provided in sklearn has a slight bug. It is not
a fatal bug as it does not affect results to a large degree, but it is still a bug.

logistic_regression.py: Platt-scaling - Basically logistic regression without regularization.
One situation where a separate platt-scaling function from the one integrated into most SVM
libraries is if you change sampling rate. In such cases, the built-in version would give wildly
misleading results. Also, this is not an efficient version of platt-scaling.

my_enir.py: ENIR - Ensemble of Near-Isotonic Regression (Naeini & al.). The version provided by the original
authors is matlab-code, hence a python-version is needed. Also, the matlab version sometimes,
although rarely, produces probabilities that are outside of [0, 1]. My version does not include
the priors used by the original authors, and hence produces slightly less well-calibrated
probabilities. Maybe I will fix this one day. This might just be the best calibration method
out there.

my_ir.py: BEIR - Ensemble of isotonic regression with model weighting. This actually produces slightly
better results than IR, but does not reach as good results as ENIR.

isotonic.py: WABIR - Weighted average of bootstrap isotonic regression. Borrowing ideas from BBQ and
ENIR, this version of isotonic regression creates an ensemble of isotonic regression models using
a bootstrap-approach and further weights these models based on likelihood.


Not yet implemented:

BBQ - Bayesian binning by quantiles. Maybe I won't implement this as it contains a rather 
problematic flaw: It turns out that with highly imbalanced datasets, the highest scoring samples
will be assigned unreasonably low probabilities.

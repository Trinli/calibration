# Calibration

"Calibration" is the process of turning e.g. machine learning predictions into probabilities.
This is particularly useful whenever the predictions are used downstream to estimate expected
values and similar.

There are a number of calibration methods. This repository contains a handful I have decided
to implement myself both for learning to understand how and why they actually work, as well
as making some comparisons.

To be added:

Isotonic regression - It turns out the version provided in sklearn has a slight bug. It is not
a fatal bug as it does not affect results to a large degree, but it is still a bug.

Platt-scaling - Basically logistic regression without no bias parameter and no regularization.
One situation where a separate platt-scaling function from the one integrated into most SVM
libraries is if you change sampling rate. In such cases, the built-in version would give wildly
misleading results. Also, this is not an efficient version of platt-scaling.

ENIR - Ensemble of Near-Isotonic Regression (Naeini & al.). The version provided by the original
authors is matlab-code, hence a python-version is needed. Also, the matlab version sometimes,
although rarely, produces probabilities that are outside of [0, 1]. My version does not include
the priors used by the original authors, and hence produces slightly less well-calibrated
probabilities. Maybe I will fix this one day. This might just be the best calibration method
out there.

BBQ - Bayesian binning by quantiles. Maybe I won't implement this as it contains a rather 
problematic flaw: It turns out that with highly imbalanced datasets, the highest scoring samples
will be assigned unreasonably low probabilities.

BEIR - Ensemble of isotonic regression with model weighting. This actually produces slightly
better results than IR, but does not reach as good results as ENIR.

**** - Isotonic regression with quality guarantees. The bins in isotonic regression should actually
be characterized by the beta-distribution, hence quality guarantees can be assigned to bins.
Whenever the quality does not reach some predefined level (i.e. credible interval not narrow enough)
a bin is merged with its neighboring bin to produce a more peaked beta-distribution and hence
better quality guarantees. The code for this one is a mess for the moment, though...

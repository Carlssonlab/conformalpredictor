
import numpy as np
import random
random.seed(10)

from bisect import bisect_right
from bisect import bisect_left
from scipy.special import softmax


def get_Nonconformity_scores(model, data):

    f = lambda x: 1-x
    
    try:
    
        nonconformity_scores = f(model.predict_proba(data))
    
    except:

        # Assuming BERT models

        _ , raw_outputs = model.predict(data.tolist())
        probabilities = softmax(raw_outputs, axis=1)
        nonconformity_scores = f(probabilities)

    return nonconformity_scores


def cali_nonconf_scores(X_calibration_data, y_calibration_data, ml_model):
    """Determine the conformity scores of the calibration data.
    Get prediction probabilities for all classes and store them if
    they belong to the true class, in separate vectors.
    """
    # calibration_alphas = get_Nonconformity_scores(ml_model, calibration_data[:,1:])
    calibration_alphas = get_Nonconformity_scores(ml_model, X_calibration_data)

    # Get number of classes 
    nr_class = len(calibration_alphas[0])

    # Iterate over all classes and retrieve calibration scores
    for c in range(nr_class):
        calibration_alpha_c = np.array([ calibration_alphas[i][c] for i in range(len(calibration_alphas)) if y_calibration_data[i] == c ])
    
        # Sorting arrays in-place without a copy (lowest to highest).
        calibration_alpha_c.sort()

        yield calibration_alpha_c


def get_CP_p_value(nonconf_score_c, calibration_alphas_c, smooth=True):
    """Returns the p-value of a sample as determined from
    the calibration set's conformity scores and the
    positive and negative conformity scores for a given sample.
    """
    # Getting indices for the conformity scores from the sorted lists.
    #  If the current alpha (conf score) would be inserted into the 
    #  sorted calibration conf scores they would be located at the
    #  following position.
    size_cal_list = len(calibration_alphas_c)
    index_p_c = bisect_left(calibration_alphas_c, nonconf_score_c)
    n_equal_or_over = size_cal_list - index_p_c

    # The relative position is outputted as the conformal
    #  prediction's p value.
    if not smooth:

        p_c = float(n_equal_or_over)/float(size_cal_list+1)

    else:

        right_index = bisect_right(calibration_alphas_c, nonconf_score_c)
        n_equal = right_index - index_p_c
        n_over = n_equal_or_over - n_equal
        p_c = (n_over + (n_equal * random.random())) / (float(size_cal_list+1))
    
    return p_c
    


def set_prediction(p0, p1, significance):
    """Determines the classification set of a sample."""

    set_prediction ='{}'

    if p0 > significance:

        set_prediction = '{0}'

        if p1 > significance: set_prediction = '{0,1}'

    elif p1 > significance: set_prediction = '{1}'

    return set_prediction
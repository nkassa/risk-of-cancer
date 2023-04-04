import numpy as np
from lab4_utils import feature_names

# Hint: Consider how to utilize np.unique()
def preprocess_data(training_inputs, testing_inputs, training_labels, testing_labels):
    processed_training_inputs, processed_testing_inputs = ([], [])
    processed_training_labels, processed_testing_labels = ([], [])
    # VVVVV YOUR CODE GOES HERE VVVVV $

    # ^^^^^ YOUR CODE GOES HERE ^^^^^ $
    return processed_training_inputs, processed_testing_inputs, processed_training_labels, processed_testing_labels



# Hint: consider how to utilize np.count_nonzero()
def naive_bayes(training_inputs, testing_inputs, training_labels, testing_labels):
    assert len(training_inputs) > 0, f"parameter training_inputs needs to be of length 0 or greater"
    assert len(testing_inputs) > 0, f"parameter testing_inputs needs to be of length 0 or greater"
    assert len(training_labels) > 0, f"parameter training_labels needs to be of length 0 or greater"
    assert len(testing_labels) > 0, f"parameter testing_labels needs to be of length 0 or greater"
    assert len(training_inputs) == len(training_labels), f"training_inputs and training_labels need to be the same length"
    assert len(testing_inputs) == len(testing_labels), f"testing_inputs and testing_labels need to be the same length"
    misclassify_rate = 0
    # VVVVV YOUR CODE GOES HERE VVVVV $

    # ^^^^^ YOUR CODE GOES HERE ^^^^^ $
    return misclassify_rate


# Hint: reuse naive_bayes to compute the misclassification rate for each fold.
def cross_validation(training_inputs, testing_inputs, training_labels, testing_labels):
    data = np.concatenate((training_inputs, testing_inputs))
    label = np.concatenate((training_labels, testing_labels))
    average_rate = 0
    # VVVVV YOUR CODE GOES HERE VVVVV $

    # VVVVV YOUR CODE GOES HERE VVVVV $
    return average_rate

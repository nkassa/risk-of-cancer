from lab4_utils import (
    load_data, accuracy_score
)
from lab4 import cross_validation, naive_bayes, preprocess_data
import numpy as np
import time


if __name__ == "__main__":
    print(f"Loading data from file...")
    start_time = time.time()
    raw_data = load_data()
    load_time = time.time() - start_time
    print(f"Data loaded - time elapsed from start: {load_time:0.9f}")
    print(f"Beginning data preprocessing and cleaning...")
    processed_training_inputs, processed_testing_inputs, processed_training_labels, processed_testing_labels =\
        preprocess_data(*raw_data)
    load_time = time.time() - start_time
    print(f"Data preprocessed - time elapsed from start: {load_time:0.9f}")
    print(f"Example training input: {processed_training_inputs[0]} - label: {processed_training_labels[0]}")
    misclassify_rate, average_misclassify_rate = 0,0

    misclassify_rate = naive_bayes(processed_training_inputs, processed_testing_inputs, processed_training_labels, processed_testing_labels)
    print(f"Misclassification Rate: {misclassify_rate:0.4f}")

    average_misclassify_rate = cross_validation(processed_training_inputs, processed_testing_inputs, processed_training_labels, processed_testing_labels)
    print(f"Average Misclassification Rate: {average_misclassify_rate:0.4f}")


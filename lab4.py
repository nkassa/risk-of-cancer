import numpy as np
import pandas as pd 
from lab4_utils import feature_names

# Hint: Consider how to utilize np.unique()
def preprocess_data(training_inputs, testing_inputs, training_labels, testing_labels):
    processed_training_inputs, processed_testing_inputs = ([], [])
    processed_training_labels, processed_testing_labels = ([], [])
    # VVVVV YOUR CODE GOES HERE VVVVV $
    # loop through the column of the training input to find the '?'
    for x in range(training_inputs.shape[1]):
    	# we are looping over the columns 
    	columns = training_inputs[:, x]
    	# unique value
    	uniqueval = np.unique(columns)
    	# number of times repeated calculated in this loop
    	# create array the size of uniqueval
    	val_cnt = np.empty_like(uniqueval)
    	for i in range(len(uniqueval)):
    		# in the array store all the values of the column
    		val = uniqueval[i]
    		val_cnt[i] = np.count_nonzero(columns == val)
    	# return indices of the maximum values (index of most repeated data)
    	most_repeated = np.argmax(val_cnt)
    	#mode
    	mode = uniqueval[most_repeated]
    	# replace '?' with mode for training  
    	# find the '?' 
    	train_question = training_inputs[:, x] == '?'
    	# replace '?' with mode
    	training_inputs[train_question , x] = mode
    	
    # loop through the column of the testing input to find the '?'
    for x in range(testing_inputs.shape[1]):
    	# we are looping over the columns 
    	columns = testing_inputs[:, x]
    	# unique value
    	uniqueval = np.unique(columns)
    	# number of times repeated calculated in this loop
    	# create array the size of uniqueval
    	val_cnt = np.empty_like(uniqueval)
    	for i in range(len(uniqueval)):
    		# in the array store all the values of the column
    		val = uniqueval[i]
    		val_cnt[i] = np.count_nonzero(columns == val)
    	# return indices of the maximum values (index of most repeated data)
    	most_repeated = np.argmax(val_cnt)
    	#mode
    	mode = uniqueval[most_repeated]
    	# replace '?' with mode for testing 
    	# find the '?' 
    	test_question = testing_inputs[:, x] == '?'
    	# replace '?' with mode
    	testing_inputs[test_question , x] = mode
    
    #map of feature names (key is name, value is the order in which they appear in feature_names)
    featurenames = {
    'age_group': 0,
    'menopause': 1,
    'tumor_size': 2,
    'inv_nodes': 3,
    'node_caps': 4,
    'deg_malig': 5,
    'side': 6,
    'quadrant': 7,
    'irradiated': 8
    }
    
    #hard coding the ordinal features based on instructions 
    ordinal = {
    'age_group':{"10-19": 1, "20-29": 2, "30-39": 3, "40-49": 4, "50-59": 5, "60-69": 6, "70-79": 7, "80-89": 8, "90-99": 9},
    'tumor_size':{"0-4":1, "5-9":2, "10-14":3, "15-19":4, "20-24":5, "25-29":6, "30-34":7, "35-39":8, "40-44":9, "45-49":10, "50-54":11, "55-59":12},
    'inv_nodes':{"0-2": 1, "3-5": 2, "6-8": 3, "9-11": 4, "12-14":5, "15-17":6, "18-20":7, "21-23":8, "24-26":9, "27-29":10, "30-32":11, "33-35":12, "36-39":13},
    'deg_malig':{1: 1,2: 2,3: 3}
    }
    
    #hard coding the categorical (treat it like ordinal)
    categorical = {
    'irradiated':{"yes": 1, "no": 0},
	 'node_caps':{"yes": 1, "no": 0},
	 'side':{"left": 0, "right": 1},
	 'quadrant': {'left_up': 0, 'left_low': 1, 'right_up': 2, 'right_low': 3, 'central': 4 },
	 'menopause': {'ge40': 0, 'lt40': 1, 'premeno': 2 }
	 }
    	
    # converting ordinal to numeric features 
    # for training and testing
    for feature_indx in ordinal:
    	val = ordinal[feature_indx]
    	name = featurenames[feature_indx]
    	training_inputs[:, name] = np.vectorize(val.get)(training_inputs[:, name])
    	testing_inputs[:, name] = np.vectorize(val.get)(testing_inputs[:, name])	
    	
    #converting categorical to numeric features
    # for training and testing 
    for feature_indx in categorical:
    	val = categorical[feature_indx]
    	name = featurenames[feature_indx]
    	training_inputs[:, name] = np.vectorize(val.get)(training_inputs[:, name])
    	testing_inputs[:, name] = np.vectorize(val.get)(testing_inputs[:, name])	
    
    processed_training_inputs = training_inputs
    processed_testing_inputs = testing_inputs
    processed_training_labels = training_labels
    processed_testing_labels = testing_labels
    #print(processed_testing_inputs)
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
    
    # count # of training labels as "no-recurrence-events"
    non_recu = np.count_nonzero(training_labels == "no-recurrence-events")
    # count # of training labels as "recurrence-events"
    recu = np.count_nonzero(training_labels == "recurrence-events")
    
    
    # Laplace smoothing (add 1 to numerator and 2 to the denominator for the two label options)
    
    # the probability of a training point in "no-recurrence-events"
    prob_non_recu = (non_recu + 1) / (len(training_labels) + 2)
    # the probability of a training point in "recurrence-events"
    prob_recu = (recu + 1) / (len(training_labels) + 2)
    
    # compute most likely probabilities using Laplace smoothing 
    most_common = {}
    # loop through column
    for f_indx in range(training_inputs.shape[1]):
    	# get values in column 
    	f_value = training_inputs[:, f_indx]
    	# get unique values
    	uniq = np.unique(f_value)
    	# initialize counters
    	non_recu_val = np.zeros(len(uniq))
    	recu_val = np.zeros(len(uniq))
    	# loop through unique values 
    	for y in range(len(uniq)):
    		value = uniq[y]
    		# count # of times value appears in dataset for non recurring 
    		non_recu_val[y] += np.count_nonzero(np.isin(f_value, value) & (training_labels == "no-recurrence-events"))
    		# count # of times value appears in dataset for recurring 
    		recu_val[y] += np.count_nonzero(np.isin(f_value, value) & (training_labels == "recurrence-events"))
    	# dict for each column 
    	most_common[f_indx] = {}
    	# loop through unique values 
    	for z in range(len(uniq)):
    		value = uniq[z]
    		# dict for unique value in each column 
    		most_common[f_indx][value] = {}
    		# Laplace smoothing for no recurrence
    		most_common[f_indx][value]["no-recurrence-events"] = (non_recu_val[z] + 1) / (non_recu + len(uniq))
    		# Laplace smoothing for recurrence
    		most_common[f_indx][value]["recurrence-events"] = (recu_val[z] + 1) / (recu + len(uniq))
    	
    
    # loop through testing inputs
    for x, (f, lab) in enumerate(zip(testing_inputs, testing_labels)):
    	# calc posterior prob for labels 
    	lab_post_non_rec, lab_post_rec = prob_non_recu, prob_recu
    	for idx in range(training_inputs.shape[1]):
    		f_val = testing_inputs[x, idx]
    		if f_val in most_common[idx]:
    			lab_post_non_rec *= most_common[idx][f_val]["no-recurrence-events"] 
    			lab_post_rec *= most_common[idx][f_val]["recurrence-events"] 
    		else:
    			lab_post_non_rec *= 1 / (non_recu + len(uniq))
    			lab_post_rec *= 1 / (recu + len(uniq))
    	# determine pred class using posterior prob
    	pre_class = "recurrence-events" if lab_post_non_rec < lab_post_rec else "no-recurrence-events"
    	# if incorrect increment by 1
    	if pre_class != lab:
    		misclassify_rate = misclassify_rate + 1	
    # calc misclassification in terms of all testing labels
    misclassify_rate /= len(testing_labels)

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

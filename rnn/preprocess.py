import tensorflow as tf
import numpy as np
from functools import reduce

def get_data(train_file, test_file):
    """
    Read and parse the train and test file line by line, then tokenize the sentences to build the train and test data separately.
    Create a vocabulary dictionary that maps all the unique tokens from your train and test data as keys to a unique integer value.
    Then vectorize your train and test data based on your vocabulary dictionary.

    :param train_file: Path to the training file.
    :param test_file: Path to the test file.
    :return: Tuple of train (1-d list or array with training words in vectorized/id form), test (1-d list or array with testing words in vectorized/id form), vocabulary (Dict containg index->word mapping)
    """

    # TODO: load and concatenate training data from training file.

    # TODO: load and concatenate testing data from testing file.

    # TODO: read in and tokenize training data

    # TODO: read in and tokenize testing data

    # TODO: return tuple of training tokens, testing tokens, and the vocab dictionary.

    train_data=open(train_file, 'r')
    test_data=open(test_file, 'r')

    train=[]
    test=[]
    vocab_dict=dict()
    curr_id=0

    for line in train_data:
        words=line.split()
        for w in words:
            if w not in vocab_dict:
                vocab_dict[w]=curr_id
                curr_id+=1
            w_id=vocab_dict[w]
            train.append(w_id)
    train=np.array(train)


    for line in test_data:
        words=line.split()
        for w in words:
            if w not in vocab_dict:
                vocab_dict[w]=curr_id
                curr_id+=1
            w_id=vocab_dict[w]
            test.append(w_id)
    test=np.array(test)

    return vocab_dict,train,test


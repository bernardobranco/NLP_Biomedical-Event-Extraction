import math
from collections import defaultdict
import statnlpbook.bio as bio
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
import loadData as ld
import statnlpbook.util as util
import string
from sklearn import svm

event_train = ld.event_train
event_dev = ld.event_dev

def sentenceToShortPath(event, sent):
    """
    Returns the path between two arguments in a sentence, where the arguments have been masked
    Args:
        sent: the sentence
    Returns:
        the path between to arguments
    """
    trigger_index = event.trigger_index
    for protein in event.sent.mentions:
        if protein['begin'] > trigger_index:
            words_in_between = ''.join(event.sent.tokens[x]['word'] for x in range(trigger_index,protein['begin']))
        elif protein['end'] < trigger_index:
            words_in_between = ''.join(event.sent.tokens[x]['word'] for x in range(protein['end'], trigger_index))
    return words_in_between


def searchForPatternsAndEntpairsByPatterns(training_patterns, testing_patterns, testing_entpairs, testing_sentences):
    testing_extractions = []
    appearing_testing_patterns = []
    appearing_testing_entpairs = []
    for i, testing_pattern in enumerate(testing_patterns):
        if testing_pattern in training_patterns: # if there is an exact match of a pattern
            testing_extractions.append(testing_sentences[i])
            appearing_testing_patterns.append(testing_pattern)
            appearing_testing_entpairs.append(testing_entpairs[i])
    return testing_extractions, appearing_testing_patterns, appearing_testing_entpairs


def searchForPatternsAndEntpairsByEntpairs(training_entpairs, testing_patterns, testing_entpairs, testing_sentences):
    testing_extractions = []
    appearing_testing_patterns = []
    appearing_testing_entpairs = []
    for i, testing_entpair in enumerate(testing_entpairs):
        if testing_entpair in training_entpairs: # if there is an exact match of an entity pair
            testing_extractions.append(testing_sentences[i])
            appearing_testing_entpairs.append(testing_entpair)
            appearing_testing_patterns.append(testing_patterns[i])
    return testing_extractions, appearing_testing_patterns, appearing_testing_entpairs

def bootstrappingExtraction(train_sents, train_entpairs, test_sents, test_entpairs, num_iter):
    """
    Given a set of patterns and entity pairs for a relation, extracts more patterns and entity pairs iteratively
    Args:
        train_sents: training sentences with arguments masked
        train_entpairs: training entity pairs
        test_sents: testing sentences with arguments masked
        test_entpairs: testing entity pairs
    Returns:
        the testing sentences which the training patterns or any of the inferred patterns appeared in
    """

    # convert training and testing sentences to short paths to obtain patterns
    train_patterns = set([sentenceToShortPath(test_sent) for test_sent in train_sents])
    test_patterns = [sentenceToShortPath(test_sent) for test_sent in test_sents]
    test_extracts = []

    # iteratively get more patterns and entity pairs
    for i in range(1, num_iter):
        print("Number extractions at iteration", str(i), ":", str(len(test_extracts)))
        print("Number patterns at iteration", str(i), ":", str(len(train_patterns)))
        print("Number entpairs at iteration", str(i), ":", str(len(train_entpairs)))
        # get more patterns and entity pairs
        test_extracts_p, ext_test_patterns_p, ext_test_entpairs_p = searchForPatternsAndEntpairsByPatterns(train_patterns, test_patterns, test_entpairs, test_sents)
        test_extracts_e, ext_test_patterns_e, ext_test_entpairs_e = searchForPatternsAndEntpairsByEntpairs(train_entpairs, test_patterns, test_entpairs, test_sents)
        # add them to the existing entity pairs for the next iteration
        train_patterns.update(ext_test_patterns_p)
        train_patterns.update(ext_test_patterns_e)
        train_entpairs.extend(ext_test_entpairs_p)
        train_entpairs.extend(ext_test_entpairs_e)
        test_extracts.extend(test_extracts_p)
        test_extracts.extend(test_extracts_e)

    return test_extracts

training_patterns = []
training_entpairs = []
for event,label_train in event_train:
    training_patterns.append(event)
    training_entpairs.append(label_train)

testing_patterns = []
testing_entpairs = []
for event,label_train in event_train:
    testing_patterns.append(event)
    testing_entpairs.append(label_train)

#test_extracts = bootstrappingExtraction(training_patterns, training_entpairs, testing_patterns, testing_entpairs, num_iter=6)
#print(test_extracts[0:3])
#print(test_extracts[-4:-1])
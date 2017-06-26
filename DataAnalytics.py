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



'''
print('------------------------------------------- Binding -----------------------------------------------')
res = viewCandidates(event_train,'Binding')
print(res)
printPretty(res)
print('------------------------------------------- Gene_expression -----------------------------------------------')
res = viewCandidates(event_train,'Gene_expression')
print(res)
printPretty(res)
print('------------------------------------------- Localization -----------------------------------------------')
res = viewCandidates(event_train,'Localization')
print(res)
printPretty(res)
print('------------------------------------------- Negative_regulation -----------------------------------------------')
res = viewCandidates(event_train,'Negative_regulation')
print(res)
printPretty(res)
print('------------------------------------------- None -----------------------------------------------')
res = viewCandidates(event_train,'None')
print(res)
printPretty(res)
print('------------------------------------------- Phosphorylation -----------------------------------------------')
res = viewCandidates(event_train,'Phosphorylation')
print(res)
printPretty(res)
print('------------------------------------------- Positive_regulation -----------------------------------------------')
res = viewCandidates(event_train,'Positive_regulation')
print(res)
printPretty(res)
print('------------------------------------------- Protein_catabolism -----------------------------------------------')
res = viewCandidates(event_train,'Protein_catabolism')
print(res)
printPretty(res)
print('------------------------------------------- Regulation -----------------------------------------------')
res = viewCandidates(event_train,'Regulation')
print(res)
printPretty(res)
print('------------------------------------------- Transcription -----------------------------------------------')
res = viewCandidates(event_train,'Transcription')
print(res)
printPretty(res)

#print(discriminatory_candidates)

#print(discriminatory_triggers)

print('------------------------------None--------------------------------------')
for x,y in event_train[:30]:
    if y == 'None':
        printSent(x)

print('-----------------------------Not None--------------------------------------')
for x,y in event_train[:100]:
    if y != 'None':
        printSent(x)
'''


def avg_trigger_reach_none(event_train):
    count = 0
    reach = 0
    for event, y in event_train:
        if y == 'None':
            reach += trigger_reach(event, event.trigger_index) - event.trigger_index
            count += 1
    return reach / count

def avg_trigger_reach_not_none(event_train):
    count = 0
    reach = 0
    for event, y in event_train:
        if y != 'None':
            reach += trigger_reach(event, event.trigger_index) - event.trigger_index
            count += 1
    return reach / count

def avg_num_words_none(event_train):
    count = 0
    reach = 0
    for event, y in event_train:
        if y == 'None':
            reach += len(event.sent.tokens)
            count += 1
    return reach / count

def avg_num_words_not_none(event_train):
    count = 0
    reach = 0
    for event, y in event_train:
        if y != 'None':
            reach += len(event.sent.tokens)
            count += 1
    return reach / count

def avg_trigger_index_none(event_train):
    count = 0
    reach = 0
    for event, y in event_train:
        if y == 'None':
            reach += event.trigger_index
            count += 1
    return reach / count

def avg_trigger_index_not_none(event_train):
    count = 0
    reach = 0
    for event, y in event_train:
        if y != 'None':
            reach += event.trigger_index
            count += 1
    return reach / count

def avg_num_depend_none(event_train):
    count = 0
    reach = 0
    for event, y in event_train:
        if y == 'None':
            reach += len(event.sent.dependencies)
            count += 1
    return reach / count

def avg_num_depend_not_none(event_train):
    count = 0
    reach = 0
    for event, y in event_train:
        if y != 'None':
            reach += len(event.sent.dependencies)
            count += 1
    return reach / count


def num_child_proteins(event,result):
    index = event.trigger_index
    proteins_spans = []
    sum_proteins = 0
    for protein in event.sent.mentions:
        proteins_spans.append((protein['begin'],protein['end']))
    for child, label in event.sent.children[index]:
        for prot in proteins_spans:
            if child in prot:
                sum_proteins += 1
    result['Sum of proteins: '] += sum_proteins

'''
for event,_ in event_train:
    result = defaultdict(float)
    num_child_proteins(event,result)
    print(result)


print('Average trigger reach for none events '+str(avg_trigger_reach_none(event_train)))
print('Average trigger reach for not none events '+str(avg_trigger_reach_not_none(event_train)))

print('Average num words for none events '+str(avg_num_words_none(event_train)))
print('Average num words for not none events '+str(avg_num_words_not_none(event_train)))

print('Average trigger index for none events '+str(avg_trigger_index_none(event_train)))
print('Average trigger index for not none events '+str(avg_trigger_index_not_none(event_train)))

print('Average num dependencies for none events '+str(avg_num_depend_none(event_train)))
print('Average num dependencies for not none events '+str(avg_num_depend_not_none(event_train)))
'''

a = 'hello'
b = 'hellogood'
print(b[len(a):])


#! SETUP 1 - DO NOT CHANGE, MOVE NOR COPY
import sys, os

import math
from collections import defaultdict
import statnlpbook.bio as bio
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
import networkx as nx
import loadData as ld
import statnlpbook.util as util
import string
import re
from sklearn import svm

event_train = ld.event_train
event_dev = ld.event_dev


stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', '\n', 'the'] + list(string.punctuation))

# <--------------------------------- Data Analytics --------------------------------->

def viewTriggerLabels(event_train,event_label):
    res_labels = defaultdict(float)
    for event, y in event_train+event_dev:
        if y == event_label:
            for child in event.sent.children[event.trigger_index]:
                res_labels[child[1]] += 1.0
    return res_labels

def viewCandidates(event_train,event_label):
    res_candidates = defaultdict(float)
    for event, y in event_train:
        if y == event_label:
            for candidate in event.argument_candidate_spans:
                candidate_word = ''.join(event.sent.tokens[x]['word'] for x in range(candidate[0], candidate[1]))
                res_candidates[candidate_word] += 1.0
    return res_candidates

def viewTriggers(event_train,event_label):
    res_trigger = defaultdict(float)
    for event, y in event_train+event_dev:
        if y == event_label:
            res_trigger[event.sent.tokens[event.trigger_index]['stem']] += 1.0
    return res_trigger

def viewProteins(event_train,event_label):
    res_proteins = defaultdict(float)
    for event, y in event_train:
        if y == event_label:
            for protein in event.sent.mentions:
                protein_word = ''.join(event.sent.tokens[x]['word'] for x in range(protein['begin'], protein['end']))
                res_proteins[protein_word] += 1.0
    return res_proteins

def viewWords(event_label):
    res_words = defaultdict(float)
    for event, y in event_train+event_dev:
        if y == event_label:
            for token in event.sent.tokens:
                if token['word'] not in stop_words:
                    res_words[token['stem']] += 1.0
    return res_words

def filter_words(words_dict,thres):
    discr_words = set()
    for key in words_dict.keys():
        if words_dict[key] > thres:
            discr_words.add(key)
    return discr_words

def printPretty(res):
    for key in res.keys():
        if res[key] > 100:
            print('Word_stem: '+key+' Count: '+str(res[key]))

def find_discr_triggers(event_train):
    labels = ['Binding','Gene_expression','Localization','Negative_regulation','None','Phosphorylation','Positive_regulation','Protein_catabolism','Regulation','Transcription']
    discr_triggers = set()
    for l in labels:
        if l != 'None':
            res = filter_words(viewTriggers(event_train, l),20)
            discr_triggers = discr_triggers | res
    return discr_triggers

def find_discr_stems():
    labels = ['Binding','Gene_expression','Localization','Negative_regulation','None','Phosphorylation','Positive_regulation','Protein_catabolism','Regulation','Transcription']
    res = set()
    for l in labels:
        res = res | filter_words(viewWords(l),20)
    return res

def find_discr_proteins(event_train):
    labels = ['Binding','Gene_expression','Localization','Negative_regulation','None','Phosphorylation','Positive_regulation','Protein_catabolism','Regulation','Transcription']
    triggers_of_labels = []
    discr_triggers = set()
    for l in labels:
        res= filter_words(viewProteins(event_train, l))
        #discr_triggers = discr_triggers^res
        discr_triggers = discr_triggers | res
    return discr_triggers

def find_discr_labels(event_train):
    labels = ['Binding','Gene_expression','Localization','Negative_regulation','None','Phosphorylation','Positive_regulation','Protein_catabolism','Regulation','Transcription']
    triggers_of_labels = []
    discr_labels = set()
    for l in labels:
        res= filter_words(viewTriggerLabels(event_train, l))
        #discr_triggers = discr_triggers^res
        discr_labels = discr_labels | res
    return discr_labels

def find_discr_candidates(event_train):
    labels = ['Binding','Gene_expression','Localization','Negative_regulation','None','Phosphorylation','Positive_regulation','Protein_catabolism','Regulation','Transcription']
    discr_candidates = set()
    for l in labels:
        res = filter_words(viewCandidates(event_train, l))
        diff = res - discr_candidates
        #print(diff)
        discr_candidates = discr_candidates | diff
    return discr_candidates

def find_discr_triggers_dict(event_train):
    labels = ['Binding','Gene_expression','Localization','Negative_regulation','None','Phosphorylation','Positive_regulation','Protein_catabolism','Regulation','Transcription']
    discr_triggers = set()
    dict_res = {}
    for l in labels:
        if l != 'None':
            res = filter_words(viewTriggers(event_train, l),10)
            dict_res[l] = res
    return dict_res

discriminatory_triggers = find_discr_triggers(event_train)
#print(discriminatory_triggers)
label_trigger_dict = find_discr_triggers_dict(event_train)
#print(label_trigger_dict)
discr_stems = find_discr_stems()
print(discr_stems)
#print(discriminatory_triggers)
#print(stop_words)
#discriminatory_candidates = find_discr_candidates(event_train)
#discriminatory_proteins = find_discr_proteins(event_train)
#discriminatory_labels = find_discr_labels(event_train)


def find_proteins(event_train,event_dev):
    proteins = set()
    for event,_ in event_train+event_dev:
        for protein in event.sent.mentions:
            for index in range(protein['begin'],protein['end']):
                proteins.add(event.sent.tokens[index]['word'])
    return proteins

proteins = find_proteins(event_train,event_dev)
#print(proteins)


# <--------------------------------- Trigger features --------------------------------->

def trigger_word_feat(event,result):
    result['trigger_word=' + event.sent.tokens[event.trigger_index]['word']] += 20.0

def trigger_stem_feat(event,result):
    result['trigger_stem=' + event.sent.tokens[event.trigger_index]['stem']] += 20.0

def trigger_pos_feat(event,result):
    result['trigger_pos=' + event.sent.tokens[event.trigger_index]['pos']] += 20.0

def child_dependency_feat(event,result):
    index = event.trigger_index
    for child, label in event.sent.children[index]:
        result["Child: " + label + "->" + event.sent.tokens[child]['word']] += 5.0

def parent_dependency_feat(event,result):
    index = event.trigger_index
    for parent, label in event.sent.parents[index]:
        result["Parent: " + label + "->" + event.sent.tokens[parent]['word']] += 5.0

def trigger_parents(event,result):
    index = event.trigger_index
    for parent, label in event.sent.parents[index]:
        result["Trigger parent: " + event.sent.tokens[parent]['word']] += 2.0

def trigger_children(event,result):
    index = event.trigger_index
    for child, label in event.sent.children[index]:
        result["Trigger children: " + event.sent.tokens[child]['word']] += 2.0

def trigger_index(event,result):
    result['Trigger index:'] += event.trigger_index / event.sent.tokens[-1]['index']

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
    result['Sum of child proteins: '] += sum_proteins

def num_parent_proteins(event,result):
    index = event.trigger_index
    proteins_spans = []
    sum_proteins = 0
    for protein in event.sent.mentions:
        proteins_spans.append((protein['begin'],protein['end']))
    for parent, label in event.sent.parents[index]:
        for prot in proteins_spans:
            if parent in prot:
                sum_proteins += 1
    result['Sum of parent proteins: '] += sum_proteins

def child_proteins(event,result):
    index = event.trigger_index
    proteins_spans = []
    for protein in event.sent.mentions:
        proteins_spans.append((protein['begin'],protein['end']))
    for child, label in event.sent.children[index]:
        for prot in proteins_spans:
            if child in prot:
                trigger = event.sent.tokens[index]['word']
                #result['Child protein label: '+label] += 1.0
                result[trigger+','+label] += 1.0

def parent_proteins(event,result):
    index = event.trigger_index
    proteins_spans = []
    for protein in event.sent.mentions:
        proteins_spans.append((protein['begin'],protein['end']))
    for parent, label in event.sent.parents[index]:
        for prot in proteins_spans:
            if parent in prot:
                trigger = event.sent.tokens[index]['word']
                #result['Parent protein label: '+label] += 1.0
                result[trigger + ',' + label] += 1.0

def trigger_child_proteins_bigram(event,result):
    index = event.trigger_index
    proteins_spans = []
    for protein in event.sent.mentions:
        proteins_spans.append((protein['begin'],protein['end']))
    for child, label in event.sent.children[index]:
        for prot in proteins_spans:
            if child in prot:
                prot_word = event.sent.tokens[child]['word']
                trigger = event.sent.tokens[index]['word']
                result[trigger+','+prot_word] += 1.0

def trigger_parent_proteins_bigram(event,result):
    index = event.trigger_index
    proteins_spans = []
    for protein in event.sent.mentions:
        proteins_spans.append((protein['begin'],protein['end']))
    for parent, label in event.sent.parents[index]:
        for prot in proteins_spans:
            if parent in prot:
                prot_word = event.sent.tokens[parent]['word']
                trigger = event.sent.tokens[index]['word']
                result[trigger+','+prot_word] += 1.0


def trigger_reach(event,relative_high):  # when first called, relative_high = trigger_index
    new_relative_high = 0
    relatives = event.sent.children[relative_high]+event.sent.parents[relative_high]
    for relative,label in relatives:
        if relative > new_relative_high:
            new_relative_high = relative
    if relative_high < new_relative_high:
        return trigger_reach(event,new_relative_high)
    else:
        return relative_high


def trigger_scope(event,result):  # when first called, relative_high = trigger_index
    reach = trigger_reach(event,event.trigger_index) - event.trigger_index
    result['Trigger scope'] += reach

def is_trigger_protein(event,result):
    trigger_word = event.sent.tokens[event.trigger_index]['word']
    if trigger_word not in stop_words:
        if trigger_word in proteins:
            result['Is trigger protein: '] = 1.0
            return
    result['Is trigger protein: '] = 0.0

def words_before_trigger(event,result):
    if event.trigger_index-3 >= 0:
        word3 = event.sent.tokens[event.trigger_index-3]['word']
        result[word3] += 2.0
        pos3 = event.sent.tokens[event.trigger_index-3]['pos']
        result[pos3] += 2.0
    if event.trigger_index-2 >= 0:
        word2 = event.sent.tokens[event.trigger_index-2]['word']
        result[word2] += 2.0
        pos2 = event.sent.tokens[event.trigger_index-2]['pos']
        result[pos2] += 2.0
    if event.trigger_index-1 >= 0:
        word1 = event.sent.tokens[event.trigger_index-1]['word']
        result[word1] += 2.0
        pos1 = event.sent.tokens[event.trigger_index-1]['pos']
        result[pos1] += 2.0


def words_after_trigger(event,result):
    if event.trigger_index+3 < len(event.sent.tokens):
        word3 = event.sent.tokens[event.trigger_index+3]['word']
        result[word3] += 2.0
        pos3 = event.sent.tokens[event.trigger_index+3]['pos']
        result[pos3] += 2.0
    if event.trigger_index + 2 < len(event.sent.tokens):
        word2 = event.sent.tokens[event.trigger_index+2]['word']
        result[word2] += 2.0
        pos2 = event.sent.tokens[event.trigger_index+2]['pos']
        result[pos2] += 2.0
    if event.trigger_index + 1 < len(event.sent.tokens):
        word1 = event.sent.tokens[event.trigger_index+1]['word']
        result[word1] += 2.0
        pos1 = event.sent.tokens[event.trigger_index+1]['pos']
        result[pos1] += 2.0



def trigger_hasNumbers(event,result):
    trigger = event.sent.tokens[event.trigger_index]['word']
    if bool(re.search(r'\d', trigger)):
        result['Trigger has digits: '] += 1.0
    else:
        result['Trigger has digits: '] += 0.0


def split_2_trigger(event,result):
    letter_sets = re.findall('..',event.sent.tokens[event.trigger_index]['word'])
    for elem in letter_sets:
        result[elem] += 5.0 # 2.0

def split_3_trigger(event,result):
    letter_sets = re.findall('...',event.sent.tokens[event.trigger_index]['word'])
    for elem in letter_sets:
        result[elem] += 5.0 # 2.0

def trigger_tail(event,result):
    trigger_stem = event.sent.tokens[event.trigger_index]['stem']
    result['Trigger tail= '+event.sent.tokens[event.trigger_index]['word'][len(trigger_stem):]] += 20.0

def is_start_trigger_capital(event,result):
    if event.sent.tokens[event.trigger_index]['word'][0].isupper():
        result['Is first letter capitalized:'] += 1.0
    else:
        result['Is first letter capitalized:'] += 0.0

def is_middle_trigger_capital(event,result):
    if any(letter.isupper() for letter in event.sent.tokens[event.trigger_index]['word'][1:]):
        result['Is middle letter capitalized:'] += 1.0
        return
    result['Is middle letter capitalized:'] += 0.0

def is_trigger_hyphenated(event,result):
    if '-' in event.sent.tokens[event.trigger_index]['word']:
        result['Is trigger hyphenated:'] += 1.0
    else:
        result['Is trigger hyphenated:'] += 0.0

def trigger_feat(event,result):
    trigger_word_feat(event, result)
    trigger_stem_feat(event, result)
    trigger_pos_feat(event, result)
    child_dependency_feat(event, result)
    parent_dependency_feat(event, result)
    #is_discriminatory(event, result)
    trigger_parents(event, result) #
    trigger_children(event, result)   #
    trigger_index(event, result)   #
    trigger_scope(event, result)   #
    num_child_proteins(event, result)
    num_parent_proteins(event, result)
    child_proteins(event, result)
    parent_proteins(event, result)
    trigger_child_proteins_bigram(event, result)    #
    trigger_parent_proteins_bigram(event, result)   #
    is_trigger_protein(event, result)    #
    words_before_trigger(event, result)
    words_after_trigger(event, result)
    trigger_tail(event, result)
    is_start_trigger_capital(event, result)
    is_middle_trigger_capital(event, result)
    is_trigger_hyphenated(event, result)
    split_2_trigger(event, result)
    split_3_trigger(event, result)
    trigger_hasNumbers(event, result)
    return result

# <------------------------------------------------------------------------------------>





# <--------------------------------- Protein features --------------------------------->

def proteins_word(event,result):
    for protein in event.sent.mentions:
        protein_word = ''.join(event.sent.tokens[x]['word'] for x in range(protein['begin'], protein['end']))
        result[protein_word] += 1.0
        #for child, label in event.sent.children[elem[0]]:
        #    result["Child: " + protein + "->" + event.sent.tokens[child]['word']] += 1.0

def proteins_and_trigger(event,result):
    for protein in event.sent.mentions:
        protein_word = ''.join(event.sent.tokens[x]['word'] for x in range(protein['begin'], protein['end']))
        result['Trigger: '+event.sent.tokens[event.trigger_index]['stem']+' Protein: '+protein_word] += 1.0
    return result

def proteins_pos(event,result):
    for protein in event.sent.mentions:
        for i in range(protein['begin'], protein['end']):
            result['protein_pos: '+event.sent.tokens[i]['pos']] += 2.0
    return result

def proteins_stem(event,result):
    for protein in event.sent.mentions:
        for i in range(protein['begin'], protein['end']):
            result['protein_stem: '+event.sent.tokens[i]['stem']] += 2.0
    return result

def proteins_word(event,result):
    for protein in event.sent.mentions:
        for i in range(protein['begin'], protein['end']):
            result['protein_word: '+event.sent.tokens[i]['word']] += 2.0
    return result

def num_prot_and_trigger(event,result):
    for protein in event.sent.mentions:
        if event.trigger_index < protein['begin']:
            result['num_words in between proteins and trigger:'] += protein['begin'] - event.trigger_index
        elif event.trigger_index > protein['end']:
            result['num_words in between proteins and trigger:'] += event.trigger_index - protein['end']
    return result

def protein_child_dependency_feat(event,result):
    for protein in event.sent.mentions:
        for index in range(protein['begin'],protein['end']):
            for child, label in event.sent.children[index]:
                result["Protein child: " + label + "->" + event.sent.tokens[child]['word']] += 2.0
                result["Protein child: " + event.sent.tokens[child]['word']] += 1.0

def protein_parent_dependency_feat(event,result):
    for protein in event.sent.mentions:
        for index in range(protein['begin'],protein['end']):
            for parent, label in event.sent.parents[index]:
                result["Protein parent: " + label + "->" + event.sent.tokens[parent]['word']] += 2.0
                result["Protein parent: " + event.sent.tokens[parent]['word']] += 1.0

def num_proteins(event,result):
    result['num of proteins: '] += len(event.sent.mentions)

def related_proteins(event,result):
    for protein in event.sent.mentions:
        for index in range(protein['begin'],protein['end']):
            for child, label in event.sent.children[index]:
                if child not in range(protein['begin'],protein['end']):
                    prot_word = event.sent.tokens[index]['word']
                    prot_child = event.sent.tokens[child]['word']
                    result[prot_word+','+prot_child] += 1.0
            for parent, label in event.sent.children[index]:
                if parent not in range(protein['begin'],protein['end']):
                    prot_word = event.sent.tokens[index]['word']
                    prot_parent = event.sent.tokens[child]['word']
                    result[prot_word+','+prot_parent] += 1.0
'''
def distance_between_proteins(event,result):
    proteins = list(event.sent.mentions)
    while len(proteins) > 1:
        protein1 = proteins[0]
        protein2 = proteins[1]
        distance = protein2['begin']-protein1['end']
        protein_word1 = ''.join(event.sent.tokens[x]['word'] for x in range(protein1['begin'], protein1['end']))
        protein_word2 = ''.join(event.sent.tokens[x]['word'] for x in range(protein2['begin'], protein2['end']))
        result[protein_word1+','+protein_word2] += distance
        proteins.pop(0)
'''

def createDependencyGraph(event):
    #dependency_graph = nx.Graph()
    dependency_graph = nx.DiGraph()
    for dependency in event.sent.dependencies:
        dependency_graph.add_edge(dependency['head'], dependency['mod'])
    return dependency_graph

def shortest_path_trigger_proteins(event,result):
    dependency_graph = createDependencyGraph(event)
    trigger_index = event.trigger_index
    if trigger_index in dependency_graph.nodes():
        for protein in event.sent.mentions:
            try:
                if protein['begin'] in dependency_graph.nodes():
                    result['shortest path between trigger and '+event.sent.tokens[protein['begin']]['word']] = nx.shortest_path_length(dependency_graph,source=trigger_index,target=protein['begin'])
            except nx.NetworkXNoPath:
                continue

def shortest_path_trigger_proteins1(event,result):
    dependency_graph = createDependencyGraph(event)
    trigger_index = event.trigger_index
    shortest_paths = []
    if trigger_index in dependency_graph.nodes():
        for protein in event.sent.mentions:
            try:
                if protein['begin'] in dependency_graph.nodes():
                    shortest_paths.append(nx.shortest_path_length(dependency_graph,source=trigger_index,target=protein['begin']))
            except nx.NetworkXNoPath:
                continue
    if shortest_paths != []:
        result['shortest path between trigger and proteins: '] += min(shortest_paths)


def shortest_path_between_proteins(event,result):
    dependency_graph = createDependencyGraph(event)
    for protein1 in event.sent.mentions:
        if protein1['begin'] in dependency_graph.nodes():
            for protein2 in event.sent.mentions:
                if protein2['begin'] in dependency_graph.nodes():
                    try:
                        shortest_path = nx.shortest_path_length(dependency_graph,source=protein1['begin'],target=protein2['begin'])
                    except nx.NetworkXNoPath:
                        continue
                    if shortest_path != 0:
                        protein_word1 = event.sent.tokens[protein1['begin']]['word']
                        protein_word2= event.sent.tokens[protein2['begin']]['word']
                        result['Shortest path between '+protein_word1+' and '+protein_word2] = shortest_path

def shortest_path_pos_between_trigger_proteins(event,result):
    dependency_graph = createDependencyGraph(event)
    trigger_index = event.trigger_index
    if trigger_index in dependency_graph.nodes():
        for protein in event.sent.mentions:
            try:
                if protein['begin'] in dependency_graph.nodes():
                    shortest_path = nx.shortest_path(dependency_graph,source=trigger_index,target=protein['begin'])
                    shortest_path_pos = ()
                    for elem in shortest_path:
                        shortest_path_pos = shortest_path_pos + (event.sent.tokens[elem]['pos'],)
                    result[str(shortest_path_pos)] += 5.0
            except nx.NetworkXNoPath:
                continue

def getLabel(event,origin,dest):
    for dependency in event.sent.dependencies:
        if dependency['head'] == origin and dependency['mod'] == dest:
            return dependency['label']

def shortest_path_labels_between_trigger_proteins(event,result):
    dependency_graph = createDependencyGraph(event)
    trigger_index = event.trigger_index
    if trigger_index in dependency_graph.nodes():
        for protein in event.sent.mentions:
            try:
                if protein['begin'] in dependency_graph.nodes():
                    shortest_path = nx.shortest_path(dependency_graph,source=trigger_index,target=protein['begin'])
                    shortest_path_labels = ()
                    for i in range(0,len(shortest_path)-1):
                        shortest_path_labels = shortest_path_labels + (getLabel(event,shortest_path[i],shortest_path[i+1]),)
                    result[str(shortest_path_labels)] += 5.0
            except nx.NetworkXNoPath:
                continue

def proteins_feat(event,result):
    proteins_pos(event,result)
    proteins_stem(event, result)
    proteins_word(event, result)
    num_prot_and_trigger(event, result)
    protein_child_dependency_feat(event, result)
    protein_parent_dependency_feat(event, result)
    num_proteins(event, result)
    #related_proteins(event, result)
    #distance_between_proteins(event, result)
    shortest_path_trigger_proteins1(event, result)
    #shortest_path_between_proteins(event, result)
    shortest_path_pos_between_trigger_proteins(event, result)
    shortest_path_labels_between_trigger_proteins(event, result)
    return result



# <------------------------------------------------------------------------------------>




# <--------------------------------- Candidate features --------------------------------->


def num_stop_words(event,result):
    for token in event.sent.tokens:
        if token['word'] in stop_words:
            result['Num stop words: '] += 1.0

def arg_word_form(event,result):
    for arg in event.argument_candidate_spans:
        for index in range(arg[0],arg[1]):
            result[event.sent.tokens[index]['word']] += 1.0

def arg_word_stem(event,result):
    for arg in event.argument_candidate_spans:
        for index in range(arg[0],arg[1]):
            result[event.sent.tokens[index]['stem']] += 1.0

def arg_word_pos(event,result):
    for arg in event.argument_candidate_spans:
        for index in range(arg[0],arg[1]):
            result[event.sent.tokens[index]['pos']] += 1.0

def arg_word_parents(event,result):
    for arg in event.argument_candidate_spans:
        for index in range(arg[0],arg[1]):
            word = event.sent.tokens[index]['word']
            for parent in event.sent.parents[index]:
                index_head,syntax = parent
                parent_token = event.sent.tokens[index_head]
                parent_word = parent_token['word']
                parent_stem = parent_token['stem']
                parent_pos = parent_token['pos']
                result[word + '_parent_info: ' +'syntax=' + syntax + ', word=' + parent_word + ', stem=' + parent_stem + ', pos=' + parent_pos] += 1.0

def arg_word_children(event,result):
    for arg in event.argument_candidate_spans:
        for index in range(arg[0],arg[1]):
            word = event.sent.tokens[index]['word']
            for child in event.sent.children[index]:
                index_child,syntax = child
                child_token = event.sent.tokens[index_child]
                child_word = child_token['word']
                child_stem = child_token['stem']
                child_pos = child_token['pos']
                result[word + '_child_info: ' +'syntax=' + syntax + ', word=' + child_word + ', stem=' + child_stem + ', pos=' + child_pos] += 1.0


def find_discriminatory_trigger_stems(event,result):
    for i in range(0,len(event.sent.tokens)):
        if i != event.trigger_index:
            stem = event.sent.tokens[i]['stem']
            if stem not in stop_words:
                if stem in discriminatory_triggers:
                    result[stem] += 10.0


def feature_candidates(event,result):
    #arg_word_form(event,result)
    #arg_word_stem(event,result)
    #arg_word_pos(event,result)
    #arg_word_parents(event, result)
    #arg_word_children(event,result)
    #num_stop_words(event, result)
    find_discriminatory_trigger_stems(event, result)
    return result

# <------------------------------------------------------------------------------------>





# <--------------------------------- Dependency features --------------------------------->

def isArgument(index,candidate_spans):
    for arg in candidate_spans:
        if index in range(arg[0],arg[1]):
            return True
    return False

def candidates(event,result):
    for arg in event.argument_candidate_spans:
        for index in range(arg[0],arg[1]):
            pos = event.sent.tokens[index]['pos']
            stem = event.sent.tokens[index]['stem']
            word = event.sent.tokens[index]['word']
            result['candidate pos: '+pos] += 1.0
            result['candidate stem: '+stem] += 1.0
            result['candidate word: '+word] += 1.0

def dependencies_candidates(event,result):
    candidate_span = event.argument_candidate_spans
    for arg in candidate_span:
        for index in range(arg[0],arg[1]):
            for child in event.sent.children[index]:
                i_child,label_child = child
                result[label_child] += 1.0
                child_token = event.sent.tokens[i_child]
                result[event.sent.tokens[index]['word']+'-> child: '+child_token['stem']] += 1.0
            for parent in event.sent.parents[index]:
                i_parent,label_parent = parent
                result[label_parent] += 1.0
                parent_token = event.sent.tokens[i_parent]
                result[event.sent.tokens[index]['word']+'-> parent: '+parent_token['stem']] += 1.0


def dependencies_of_candidates1(event,result):
    candidate_spans = event.argument_candidate_spans
    for dependency in event.sent.dependencies:
        if isArgument(dependency['head'],candidate_spans) or isArgument(dependency['mod'],candidate_spans):
            head = event.sent.tokens[dependency['head']]['word']
            label = dependency['label']
            mod = event.sent.tokens[dependency['mod']]['word']
            result['head: ' + head + ', label: ' + label + ', mod: ' + mod] += 1.0
            result['head: ' + head + ', mod: ' + mod] += 1.0

def dependencies_of_candidates(event,result):
    for arg in event.argument_candidate_spans:
        for index in range(arg[0],arg[1]):
            for dependency in event.sent.dependencies:
                if dependency['head'] == index:
                    head = event.sent.tokens[index]['word']
                    label = dependency['label']
                    mod = event.sent.tokens[dependency['mod']]['word']
                    result['head: ' + head + ', label: '+label+', mod: '+mod] += 1.0
                    result['head: ' + head + ', mod: ' + mod] += 1.0
                #if dependency['mod'] == index:
                #    mod = event.sent.tokens[index]['word']
                #    label = dependency['label']
                #    head = event.sent.tokens[dependency['head']]['word']
                #    result['head: '+head+', label: '+label+', mod: '+mod] += 1.0


def words_in_between_prot_and_candid(event,result):
    for arg in event.argument_candidate_spans:
        arg_words = ''.join(event.sent.tokens[i]['word'] for i in range(arg[0],arg[1]))
        proteins = event.sent.mentions
        for protein in proteins:
            protein_str = ''.join(event.sent.tokens[x]['word'] for x in range(protein['begin'], protein['end']))
            if arg[0] in range(protein['begin'],protein['end']):
                continue
            elif arg[1] < protein['begin'] and protein['begin']-arg[1] <= 5:
                words = ''.join(event.sent.tokens[j]['word'] for j in range(arg[1],protein['begin']))
                result['candidate: ' + arg_words + ', protein: ' + protein_str + ', words in between: ' + words] += 1.0
            elif protein['end'] < arg[0] and arg[0]-protein['end'] <= 5:
                words = ''.join(event.sent.tokens[j]['word'] for j in range(protein['end'],arg[0]))
                result['candidate: ' + arg_words + ', protein: ' + protein_str + ', words in between: ' + words] += 1.0


def num_arg_candidates(event,result):
    result['num of candidates'] += len(event.argument_candidate_spans)


def num_dependenies(event,result):
    result['num of dependencies'] += len(event.sent.dependencies)

def num_of_diff_labels(event,result):
    labels = set()
    for dependency in event.sent.dependencies:
        labels.add(dependency['label'])
    result['num of different labels:'] += len(labels)

def dependency_labels(event,result):
    for dependency in event.sent.dependencies:
        result[dependency['label']] += 1.0

def prep_dependencies(event,result):
    for dependency in event.sent.dependencies:
        if 'prep' in dependency['label']:
            head_word = event.sent.tokens[dependency['head']]['word']
            mod_word = event.sent.tokens[dependency['mod']]['word']
            result['head: ' + head_word + ' mod: ' + mod_word + ' label: ' + dependency['label']] += 1.0

def feature_dependencies(event,result):
    words_in_between_prot_and_candid(event, result)
    #dependencies_candidates(event, result)
    num_arg_candidates(event,result)
    #candidates(event, result)
    dependencies_of_candidates1(event,result)
    #prep_dependencies(event, result)
    #num_dependenies(event, result)
    #num_of_diff_labels(event, result)
    #dependency_labels(event, result)
    return result

# <------------------------------------------------------------------------------------>





# <--------------------------------- Feature vector ----------------------------------->
def feature_vector(event):
    result = defaultdict(float)
    # Trigger features:
    trigger_feat(event, result)

    #Proteins:
    proteins_feat(event, result)

    #Feature Candidates_
    feature_candidates(event,result)

    # Feature dependencies:
    feature_dependencies(event,result)
    # Bias
    #result['Bias'] += 1.0
    return result

# <------------------------------------------------------------------------------------>

def show_most_informative_features(vectorizer, clf, n=20):
    feature_names = vectorizer.get_feature_names()
    for i in range(0,10):
        print('<-------------------------------------Label--------------------------------------------->')
        coefs_with_fns = sorted(zip(clf.coef_[i], feature_names))
        top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
        for (coef_1, fn_1), (coef_2, fn_2) in top:
            print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))



# <--------------------------------- Logistic Regression ----------------------------------->

# converts labels into integers, and vice versa, needed by scikit-learn.
label_encoder = LabelEncoder()
# encodes feature dictionaries as numpy vectors, needed by scikit-learn.
vectorizer = DictVectorizer()

def train_log_regr_model(event_train,c):
    # converting training data into a representation understood by scikit-learn
    train_event_x = vectorizer.fit_transform([feature_vector(x) for x,_ in event_train])
    train_event_y = label_encoder.fit_transform([y for _,y in event_train])
    # Create and train the model. Feel free to experiment with other parameters and learners.
    lr = LogisticRegression(C=c, class_weight='balanced')
    lr.fit(train_event_x, train_event_y)
    # show most common features
    #show_most_informative_features(vectorizer, lr)
    #clf = svm.LinearSVC()
    #clf.fit(train_event_x, train_event_y)
    return lr

def predict_labels_using_model(lr,event_candidates):
    event_x = vectorizer.transform([feature_vector(e) for e in event_candidates])
    event_y = label_encoder.inverse_transform(lr.predict(event_x))
    return event_y

def train_model(event_train,event_test):
    C_range = np.linspace(0.1, 1, 10)
    C_range = [0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10]
    accuracy = []
    for c in C_range:
        lr = train_log_regr_model(event_train,c)
        event_test_guess = predict_labels_using_model(lr,[x for x,_ in event_test[:]])
        cm_test = bio.create_confusion_matrix(event_test,event_test_guess)
        accuracy.append(bio.evaluate(cm_test)[2]) # This is the F1 score
        print('C=',c, ' Accuracy=',bio.evaluate(cm_test)[2])
    plt.plot(C_range,accuracy)
    plt.show()
    print(accuracy)


# <------------------------------------------------------------------------------------>


event_corpus = event_train+event_dev

event_train = event_corpus[:len(event_corpus)//4 * 2]+event_corpus[len(event_corpus)//4 * 3:]
event_dev = event_corpus[len(event_corpus)//4 * 2:len(event_corpus)//4 * 3]

def printSent(event):
    sentence = ''
    for e in event.sent.tokens:
        sentence += e['word'] + ' '
    print(sentence)

#event_candidate, label_event = event_train[0]
#printSent(event_candidate)
#result = filter_tokens(event_candidate)
#words_in_between_prot_and_candid(event_candidate,result)
#print(result)

'''
for x,_ in event_train:
    res = defaultdict(float)
    is_trigger_protein(x,res)
    if res['Is trigger protein: '] != 0.0:
        print(res)
'''

train_model(event_train,event_dev)
#lr = train_log_regr_model(event_train,0.3)
#event_dev_guess = predict_labels_using_model(lr,[x for x,_ in event_dev[:]])
#cm_dev = bio.create_confusion_matrix(event_dev,event_dev_guess)
#print(bio.full_evaluation_table(cm_dev))
#print('Accuracy: ', bio.evaluate(cm_dev)[2])


#util.plot_confusion_matrix_dict(cm_dev,90, outside_label="None")

#errors = bio.find_errors("Binding","None", event_dev, event_dev_guess)

#for error in errors:
#    printSent(error[0])
#    print('<------------------------------>')


#print(errors[0][0].sent)

#bio.show_event_error(*errors[0])




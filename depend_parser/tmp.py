from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import keras

from extract_training_data import FeatureExtractor, State

'''
words = ["And", "they", "plan", "to", "buy", "more", "today", "."]
pos = ["CC", "PRP", "VBP", "TO", "VB", "JJR", "NN", "."]
state = State(range(1,len(words)))
state.stack.append(0) 

WORD_VOCAB_FILE = 'data/words.vocab'
POS_VOCAB_FILE = 'data/pos.vocab'

try:
    word_vocab_f = open(WORD_VOCAB_FILE,'r')
    pos_vocab_f = open(POS_VOCAB_FILE,'r') 
except FileNotFoundError:
    print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
    sys.exit(1) 

extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)

model = keras.models.load_model("/Users/apple/Desktop/semester_1/5.nlp/hw/hw3/data/model.h5")



features = extractor.get_input_representation(words, pos, state)
print(features)

soft_acts = model.predict(features.reshape(-1,6))
print(soft_acts)
'''

'''
myList = [1, 2, 3, 100, 5]


ind = [i[0] for i in sorted(enumerate(myList), key=lambda x:x[1], reverse = True)]
print(ind)
'''


model = keras.models.load_model("/Users/apple/Desktop/semester_1/5.nlp/hw/hw3/data/model.h5")

WORD_VOCAB_FILE = 'data/words.vocab'
POS_VOCAB_FILE = 'data/pos.vocab'
try:
    word_vocab_f = open(WORD_VOCAB_FILE,'r')
    pos_vocab_f = open(POS_VOCAB_FILE,'r') 
except FileNotFoundError:
    print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
    sys.exit(1) 
extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)

words = ["And", "they", "plan", "to", "buy", "more", "today", "."]
pos = ["CC", "PRP", "VBP", "TO", "VB", "JJR", "NN", "."]
state = State(range(1,len(words)))
state.stack.append(0) 

dep_relations = ['tmod', 'vmod', 'csubjpass', 'rcmod', 'ccomp', 'poss', 'parataxis', 'appos', 'dep', 'iobj', 'pobj', 'mwe', 'quantmod', 'acomp', 'number', 'csubj', 'root', 'auxpass', 'prep', 'mark', 'expl', 'cc', 'npadvmod', 'prt', 'nsubj', 'advmod', 'conj', 'advcl', 'punct', 'aux', 'pcomp', 'discourse', 'nsubjpass', 'predet', 'cop', 'possessive', 'nn', 'xcomp', 'preconj', 'num', 'amod', 'dobj', 'neg','dt','det']

while state.buffer: 
    #pass
    # TODO: Write the body of this loop for part 4 
    features = extractor.get_input_representation(words, pos, state)
    soft_acts = model.predict(features.reshape(-1,6))
    sort_ind = [i[0] for i in sorted(enumerate(soft_acts[0]), key=lambda x:x[1], reverse = True)]

    for i in sort_ind:

        if i == 90:
            if len(state.buffer) > 1:
                print("01")
                state.shift()
                break
            elif state.stack == []:
                print("02")
                state.shift()
                break
            else:
                print(1)
                continue


        elif i >= 45 and i < 90:
            if state.stack == []:
                print(1)
                continue
            else:
                print("03")
                state.right_arc(dep_relations[i-45])
                break

        else:
            if state.stack == []:
                print(1)
                continue
            elif len(state.stack) == 1:
                print(1)
                continue
            else:
                print("04")
                state.left_arc(dep_relations[i])
                break
   

'''
result = DependencyStructure()

for p,c,r in state.deps: 
    result.add_deprel(DependencyEdge(c, words[c], pos[c],p, r))

print(result)
'''














from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import keras

from extract_training_data import FeatureExtractor, State

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        
        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1,len(words)))
        state.stack.append(0) 

        dep_relations = ['tmod', 'vmod', 'csubjpass', 'rcmod', 'ccomp', 'poss', 'parataxis', 'appos', 'dep', 'iobj', 'pobj', 'mwe', 'quantmod', 'acomp', 'number', 'csubj', 'root', 'auxpass', 'prep', 'mark', 'expl', 'cc', 'npadvmod', 'prt', 'nsubj', 'advmod', 'conj', 'advcl', 'punct', 'aux', 'pcomp', 'discourse', 'nsubjpass', 'predet', 'cop', 'possessive', 'nn', 'xcomp', 'preconj', 'num', 'amod', 'dobj', 'neg','dt','det']

        while state.buffer: 
            #pass
            # TODO: Write the body of this loop for part 4 
            features = self.extractor.get_input_representation(words, pos, state)
            soft_acts = self.model.predict(features.reshape(-1,6))
            sort_ind = [i[0] for i in sorted(enumerate(soft_acts[0]), key=lambda x:x[1], reverse = True)]

            for i in sort_ind:
                if i == 90:
                    if len(state.buffer) > 1:
                        state.shift()
                        break
                    elif state.stack == []:
                        state.shift()
                        break
                    else:
                        continue

                elif i >= 45 and i < 90:
                    if state.stack == []:
                        continue
                    else:
                        state.right_arc(self.output_labels[i-45])
                        break

                else:
                    if state.stack == []:
                        continue
                    elif len(state.stack) == 1:
                        continue
                    else:
                        state.left_arc(self.output_labels[i])  
                        break               

        result = DependencyStructure()
        for p,c,r in state.deps: 
            result.add_deprel(DependencyEdge(c, words[c], pos[c],p, r))
        return result 
        

if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    #parser = Parser(extractor, sys.argv[1])
    parser = Parser(extractor, "/Users/apple/Desktop/semester_1/5.nlp/hw/hw3/data/model.h5")

    #with open(sys.argv[2],'r') as in_file: 
    with open("/Users/apple/Desktop/semester_1/5.nlp/hw/hw3/data/dev.conll",'r') as in_file: 
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
        

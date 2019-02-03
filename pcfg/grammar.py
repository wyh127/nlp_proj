"""
COMS W4705 - Natural Language Processing - Fall 2018
Homework 2 - Parsing with Context Free Grammars 
Daniel Bauer
"""

import sys
from collections import defaultdict
from math import fsum

class Pcfg(object): 
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file): 
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None 
        self.read_rules(grammar_file)      
 
    def read_rules(self,grammar_file):
        
        for line in grammar_file: 
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line: 
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else: 
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()
                    
     
    def parse_rule(self,rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";",1) 
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False. 
        """
        # TODO, Part 1
        for keys in self.lhs_to_rules.keys():
        	prob = []
        	for item in self.lhs_to_rules[keys]:
        		prob.append(item[2])
        	if((fsum(prob)-1) < 0.00000000001):
        		return True
        	else: 
        		return False

if __name__ == "__main__":
	# with open(sys.argv[1],'r') as grammar_file:
    with open('/Users/apple/Desktop/semester_1/5.nlp/hw/hw2/atis3.pcfg','r') as grammar_file:
        grammar = Pcfg(grammar_file)

    # test valid PCFG in CNF
    print(grammar.verify_grammar())

    toks = ['flights', 'from', 'miami', 'to', 'cleveland', '.'] 
    l = len(toks)-1


    print(grammar.rhs_to_rules[('flights',)])

   

    print(grammar.lhs_to_rules[('NP')])




















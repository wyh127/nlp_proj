"""
COMS W4705 - Natural Language Processing - Fall 2018
Homework 2 - Parsing with Context Free Grammars 
Daniel Bauer
"""
import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg

### Use the following two functions to check the format of your data structures in part 3 ###
def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and \
          isinstance(split[0], int)  and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str): # Leaf nodes may be strings
                continue 
            if not isinstance(bps, tuple):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(bps))
                return False
            if len(bps) != 2:
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(bps))
                return False
            for bp in bps: 
                if not isinstance(bp, tuple) or len(bp)!=3:
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(bp))
                    return False
    return True

def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True



class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar): 
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar

    def is_in_language(self, tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        # TODO, part 2
        # initilization

        # record word number
        l = len(tokens)-1
        Pi = [[[] for j in range(l+1)] for i in range(l+1)] # 'multiply' is dangerous

        # initlization
        for i in range(l):
            if self.grammar.rhs_to_rules[(tokens[i],)] != []:
                for item in self.grammar.rhs_to_rules[(tokens[i],)]:
                    Pi[i][i+1] += [item[0]]

        # CKY algoritm core part
        for m in range(2, l+1):
            for i in range(l+1-m):
                j = i+m
                for k in range(i+1, j):
                    if (Pi[i][k] != []) and (Pi[k][j] != []):
                        for B in Pi[i][k]:
                            for C in Pi[k][j]:
                                BC = (B, C)
                                if self.grammar.rhs_to_rules[BC] != []:
                                    for A in self.grammar.rhs_to_rules[BC]:
                                        if A[0] not in Pi[i][j]:
                                            Pi[i][j] += [A[0]]

        # check 
        if Pi[0][l] != []:
            for item in Pi[0][l]:
                if self.grammar.rhs_to_rules[(item, 'PUN')] != []: 
                    for head in self.grammar.rhs_to_rules[(item, 'PUN')]:
                        # make sure compatible with different start symbols
                        if head[0] == self.grammar.startsymbol:
                            return True

        return False

       
    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """
        # TODO, part 3

        table = {}
        probs = {}
        l = len(tokens)-1

        # table initlization 
        for i in range(l):
            tmp = {}
            tmp1 = {}
            if self.grammar.rhs_to_rules[(tokens[i],)] != []:
                for item in self.grammar.rhs_to_rules[(tokens[i],)]:
                    tmp[item[0]] = (item[1][0])
                    tmp1[item[0]] = math.log(item[2])
            table[(i, i+1)] = tmp
            probs[(i, i+1)] = tmp1
    

        # CKY algoritm core part
        for m in range(2, l+1):
            for i in range(l+1-m):
                j = i+m
                tmp = {}
                tmp1 = {}
                for k in range(i+1, j):
                    try:
                        if (table[(i, k)] != {}) and (table[(k, j)] != {}):
                            for B in table[(i, k)].keys():
                                for C in table[(k, j)].keys():
                                    BC = (B, C)
                                    if self.grammar.rhs_to_rules[BC] != []:
                                        for A in self.grammar.rhs_to_rules[BC]:
                                            if A[0] in tmp.keys():
                                                if math.log(A[2]) > tmp1[A[0]]:
                                                    tmp[A[0]] = ((B, i, k), (C, k, j))
                                                    tmp1[A[0]] = math.log(A[2])
                                            else:
                                                tmp[A[0]] = ((B, i, k), (C, k, j))
                                                tmp1[A[0]] = math.log(A[2])
                    except:
                        pass
                table[(i, j)] = tmp
                probs[(i, j)] = tmp1
                        
        # check 
        if table[(0, l)] != {}:
            tmp = {}
            tmp1 = {}
            prob = -math.inf
            for item in table[(0, l)].keys():
                if self.grammar.rhs_to_rules[(item, 'PUN')] != []: 
                    for head in self.grammar.rhs_to_rules[(item, 'PUN')]:
                        # make sure compatible with different start symbols
                        if head[0] == self.grammar.startsymbol and math.log(head[2]) > prob:
                            tmp[head[0]] = ((item, 0, l), ('PUN', l, l+1))
                            tmp1[head[0]] = math.log(head[2])
                            prob = math.log(head[2])
            table[(0, l+1)] = tmp
            probs[(0, l+1)] = tmp1

        table[(l, l+1)] = {'PUN': '.'}

        return table, probs



def get_tree(table, i,j, nt): 
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    # TODO: Part 4

    
    if j == i+1:
        root = table[(i, j)][nt]
        res = (nt, root)
        return res
    else:
        try:
            root = table[(i, j)][nt]
            t1 = root[0]
            t2 = root[1]
            res = (nt, get_tree(table, t1[1], t1[2], t1[0]), get_tree(table, t2[1], t2[2], t2[0]))
            return res 
        except:
            pass
    
    
 
       
if __name__ == "__main__":
    
    with open('/Users/apple/Desktop/semester_1/5.nlp/hw/hw2/atis3.pcfg','r') as grammar_file: 
        grammar = Pcfg(grammar_file) 
        parser = CkyParser(grammar)
        toks =  ['with', 'the', 'least', 'expensive', 'fare', '.']

        #print(parser.is_in_language(toks))
        table, probs = parser.parse_with_backpointers(toks)
        print(get_tree(table, 0, len(toks), grammar.startsymbol))

        #assert check_table_format(table)
        #assert check_probs_format(probs)





        

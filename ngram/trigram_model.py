import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Fall 2018
Homework 1 - Programming Component: Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """
    new_seq = ["START" for i in range(n-1)] + sequence + ["STOP"]
    res = list()
    for i in range(len(new_seq)-n+1):
        temp = tuple(new_seq[i: i+n])
        res.append(temp)
    return res


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")

        self.V = len(self.lexicon)
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)
        self.sum = sum(self.unigramcounts.values())

        generator = corpus_reader(corpusfile)
        self.len = len(list(generator))

        # generator = corpus_reader(corpusfile)
        # self.perp = self.perplexity(generator)


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = defaultdict(int) # might want to use defaultdict or Counter instead
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)

        ##Your code here
        for sentence in corpus:
            for item in get_ngrams(sentence, 1):
                self.unigramcounts[item] += 1

            for item in get_ngrams(sentence, 2):
                self.bigramcounts[item] += 1

            for item in get_ngrams(sentence, 3):
                self.trigramcounts[item] += 1
        

    def raw_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        if trigram[0:2] == ('START', 'START'):
        	return self.trigramcounts[trigram] / (self.len / self.sum)
        else:
        	return (self.trigramcounts[trigram]+1) / (self.bigramcounts[trigram[0:2]]+self.V)
        # return (self.trigramcounts[trigram]+1) / (self.bigramcounts[trigram[0:2]]+self.V)
        	

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        if bigram[0:1] == ('START', ):
            return self.bigramcounts[bigram] / (self.len / self.sum)
        else:
        	return (1+self.bigramcounts[bigram]) / (self.V+self.unigramcounts[bigram[0:1]])
        # return (1+self.bigramcounts[bigram]) / (self.V+self.unigramcounts[bigram[0:1]])

    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """
        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  
        return self.unigramcounts[unigram] / self.sum

    def generate_sentence(self, t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        return lambda1 * self.raw_trigram_probability(trigram) + lambda2 * self.raw_bigram_probability(trigram[0:2]) + lambda3 * self.raw_unigram_probability(trigram[0:1])
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        res = 0
        temp = get_ngrams(sentence, 3)
        for item in temp:
        	p = self.smoothed_trigram_probability(item)
        	if p > 0:
        		res = res + math.log2(p)
        return res

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        l = 0
        for sentence in corpus:
        	l += self.sentence_logprob(sentence)
        l = l / self.sum
        return 2 ** (-l)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            # .. 
            qq = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            total += 1
            if pp < qq:
            	correct += 1

    
        for f in os.listdir(testdir2):
            pp = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            # .. 
            qq = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            total += 1
            if pp < qq:
            	correct += 1

        return correct / total

if __name__ == "__main__":

    # model = TrigramModel(sys.argv[1]) 
    model = TrigramModel("/Users/apple/Desktop/ColumbiaUniversity/1/nlp/hw/hw1/hw1_data/brown_train.txt")


    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
    # Testing perplexity: 
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)


    # Essay scoring experiment: 
    acc = essay_scoring_experiment("/Users/apple/Desktop/ColumbiaUniversity/1/nlp/hw/hw1/hw1_data/ets_toefl_data/train_high.txt", "/Users/apple/Desktop/ColumbiaUniversity/1/nlp/hw/hw1/hw1_data/ets_toefl_data/train_low.txt", "/Users/apple/Desktop/ColumbiaUniversity/1/nlp/hw/hw1/hw1_data/ets_toefl_data/test_high", "/Users/apple/Desktop/ColumbiaUniversity/1/nlp/hw/hw1/hw1_data/ets_toefl_data/test_low")
    print(acc)


# -*- coding: utf-8 -*-
"""
basic_sentiment_analysis
~~~~~~~~~~~~~~~~~~~~~~~~

This module contains the code and examples described in 
http://fjavieralba.com/basic-sentiment-analysis-with-python.html

"""

##### Import the necessary modules
from pprint import pprint
import nltk
import yaml
import sys
import os
import re
import pandas as pd
import json

class Splitter(object):

    def __init__(self):
        self.nltk_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.nltk_tokenizer = nltk.tokenize.TreebankWordTokenizer()

    def split(self, text):
        """
        input format: a paragraph of text
        output format: a list of lists of words.
            e.g.: [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'one']]
        """
        sentences = self.nltk_splitter.tokenize(text.decode('utf-8'))
        tokenized_sentences = [self.nltk_tokenizer.tokenize(sent) for sent in sentences]
        return tokenized_sentences


class POSTagger(object):

    def __init__(self):
        pass
        
    def pos_tag(self, sentences):
        """
        input format: list of lists of words
            e.g.: [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'one']]
        output format: list of lists of tagged tokens. Each tagged tokens has a
        form, a lemma, and a list of tags
            e.g: [[('this', 'this', ['DT']), ('is', 'be', ['VB']), ('a', 'a', ['DT']), ('sentence', 'sentence', ['NN'])],
                    [('this', 'this', ['DT']), ('is', 'be', ['VB']), ('another', 'another', ['DT']), ('one', 'one', ['CARD'])]]
        """

        pos = [nltk.pos_tag(sentence) for sentence in sentences]
        #adapt format
        pos = [[(word, word, [postag]) for (word, postag) in sentence] for sentence in pos]
       
        return pos

class DictionaryTagger(object):

    def __init__(self, dictionary_paths):
        files = [open(path, 'r') for path in dictionary_paths]
        dictionaries = [yaml.load(dict_file) for dict_file in files]
        for d in dictionaries:
            for k, v in d.items():
                d[str(k)] = v
        map(lambda x: x.close(), files)
        self.dictionary = {}
        self.max_key_size = 0
        for curr_dict in dictionaries:
            for key in curr_dict:
                key = str(key)
                if key in self.dictionary:
                    self.dictionary[key].extend(curr_dict[key])
                else:
                    self.dictionary[key] = curr_dict[key]
                    self.max_key_size = max(self.max_key_size, len(key))

    def tag(self, postagged_sentences):
        return [self.tag_sentence(sentence) for sentence in postagged_sentences]

    def tag_sentence(self, sentence, tag_with_lemmas=False):
        """
        the result is only one tagging of all the possible ones.
        The resulting tagging is determined by these two priority rules:
            - longest matches have higher priority
            - search is made from left to right
        """
        tag_sentence = []
        N = len(sentence)
        if self.max_key_size == 0:
            self.max_key_size = N
        i = 0
        while (i < N):
            j = min(i + self.max_key_size, N) #avoid overflow
            tagged = False
            while (j > i):
                expression_form = ' '.join([word[0] for word in sentence[i:j]]).lower()
                expression_lemma = ' '.join([word[1] for word in sentence[i:j]]).lower()

                if tag_with_lemmas:
                    literal = expression_lemma
                else:
                    literal = expression_form
                """ EXTRA CODE for part-of-speech test (JJ, NN) """
                expression_pos= ' '.join(word[2])
                
                if literal in self.dictionary:
                    is_single_token = j - i == 1
                    original_position = i
                    i = j
                    taggings = [tag for tag in self.dictionary[literal]]
                    tagged_expression = (expression_form, expression_lemma, taggings)
                    if is_single_token: #if the tagged literal is a single token, conserve its previous taggings:
                        original_token_tagging = sentence[original_position][2]
                        tagged_expression[2].extend(original_token_tagging)
                    tag_sentence.append(tagged_expression)
                    tagged = True
                else:
                    j = j - 1
            if not tagged:
                tag_sentence.append(sentence[i])
                i += 1
        return tag_sentence

def value_of(sentiment):
    if sentiment == 'positive': return 1
    if sentiment == 'negative': return -1
    return 0

def sentence_score(sentence_tokens, previous_token, acum_score):    
    if not sentence_tokens:
        return acum_score
    else:
        current_token = sentence_tokens[0]
        tags = current_token[2]
        token_score = sum([value_of(tag) for tag in tags])
        if previous_token is not None:
            previous_tags = previous_token[2]
            if 'inc' in previous_tags:
                token_score *= 2.0
            elif 'dec' in previous_tags:
                token_score /= 2.0
            elif 'inv' in previous_tags:
                token_score *= -1.0
        return sentence_score(sentence_tokens[1:], current_token, acum_score + token_score)

def sentiment_score(review):
    return sum([sentence_score(sentence, None, 0.0) for sentence in review])


###########################################################################################################
###############  From this point the forked code is adjusted/created for this project      ################
###########################################################################################################


def analyze_text(text):

    splitter = Splitter()
    postagger = POSTagger()
    dicttagger = DictionaryTagger([ 'dicts/positive.yml', 'dicts/negative.yml', 'dicts/inc.yml', 'dicts/dec.yml', 'dicts/inv.yml','dicts/library.yml','dicts/dutch.yml'])
    
    splitted_sentences = splitter.split(text)
    pos_tagged_sentences = postagger.pos_tag(splitted_sentences)
    dict_tagged_sentences = dicttagger.tag(pos_tagged_sentences)
    score = sentiment_score(dict_tagged_sentences)
    return score

if __name__ == '__main__':


    ## Read in data
    dataset = pd.read_csv('bol-book-reviews.csv', header=None, error_bad_lines=False)
    dataset.columns = ['UID','Book Title','Grade','Summary Title','Summary']   
    item = 0
    unique_titels = [] 
    ## keep tracks of the lists to make sure that the reviews wont be done multiple times
    bekeken = [] 
    niet_bekeken = []
    missing = ok = wrong = 0
    missing_pos = ok_pos = wrong_pos = 0
    missing_neg = ok_neg = wrong_neg = 0

    ## loop trough all the books an make sure no duplicates are created
    for titel in dataset["Book Title"]:
        if titel not in unique_titels:
            unique_titels.append(titel)

    
    titel = ''
    BID = 0

    for x in unique_titels:

        counter = 0
        gemiddeld = 0.0
        hoogste = (00 ,'')
        laagste = (00 ,'')
        max_score = -9999
        min_score = 9999

        ## if a book is not checked yet and the title is in the database -> perform sentiment analysis
        if x not in bekeken:

            for boek in dataset["Book Title"]:

                if boek == x:
                    titel = boek
                    review = dataset['Summary'][item]
                    if dataset['Grade'][item] in [1.0,2.0,3.0]:
                        gold_cats = ['neg']
                    else:
                        gold_cats = ['pos']
                    my_score = analyze_text(review)
                    
                    """ EXTRA CODE to collect numbers for precision/ recall calculations """
                    my_label = None
                    if my_score >= 1:
                        my_label = 'pos'
                    elif my_score <= -1:
                        my_label = 'neg'

                    my_res = 'Missing'          
                    if my_label is None:
                        missing += 1
                    else:
                        if my_label in gold_cats:
                            ok += 1
                            my_res = 'OK'
                        else:
                            wrong += 1
                            my_res = 'WRONG'

                    
                    if 'pos' in gold_cats:      
                        if my_label is None:
                            missing_pos += 1
                        else:
                            if my_label in gold_cats:
                                ok_pos += 1
                            else:
                                wrong_pos += 1

                    if 'neg' in gold_cats:      
                        if my_label is None:
                            missing_neg += 1
                        else:
                            if my_label in gold_cats:
                                ok_neg += 1
                            else:
                                wrong_neg += 1

                    ## Keeps track of what the most positive and most negative reviews are
                    if my_score > max_score and gold_cats == ['pos'] and my_res == 'OK':
                        max_score = my_score
                        hoogste = (item, dataset["Summary"][item])
                    elif my_score < min_score and gold_cats == ['neg'] and my_res == 'OK':
                        min_score = my_score
                        laagste = (item, dataset["Summary"][item])

                    gemiddeld += dataset["Grade"][item]
                    item += 1
                    counter += 1

            bekeken.append(x)
            print titel, BID
            rating = gemiddeld/counter

            ## Write the results to a json file
            with open("jsontest","a") as outfile:
                json.dump({'ID':BID, 'titel':titel, 'rating':rating, 'hoogste':hoogste, 'laagste':laagste}, outfile, indent=4)
                outfile.write(os.linesep)

            BID += 1            
        
  
# Steven Rud
# CS114B Spring 2023 Homework 4 
# Part-of-speech Tagging with Structured Perceptrons

import os
import numpy as np
from collections import defaultdict
from random import Random

class POSTagger():

    def __init__(self):
        # for testing with the toy corpus from the lab 7 exercise
        # self.tag_dict = {'nn': 0, 'vb': 1, 'dt': 2}
        # self.word_dict = {'Alice': 0, 'admired': 1, 'Dorothy': 2, 'every': 3,
        #                   'dwarf': 4, 'cheered': 5}
        self.tag_dict = dict()
        self.word_dict = dict()
        self.initial = np.array([-0.3, -0.7, 0.3])
        self.transition = np.array([[-0.7, 0.3, -0.3],
                                    [-0.3, -0.7, 0.3],
                                    [0.3, -0.3, -0.7]])
        self.emission = np.array([[-0.3, -0.7, 0.3],
                                  [0.3, -0.3, -0.7],
                                  [-0.3, 0.3, -0.7],
                                  [-0.7, -0.3, 0.3],
                                  [0.3, -0.7, -0.3],
                                  [-0.7, 0.3, -0.3]])
        # should raise an IndexError; if you come across an unknown word, you
        # should treat the emission scores for that word as 0
        self.unk_index = -1

    '''
    Fills in self.tag_dict and self.word_dict, based on the training data.
    '''
    def make_dicts(self, train_set):
        # iterate over training documents
        for root, dirs, files in os.walk(train_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    lines = f.read().rsplit()
                    for l in lines:
                        word,tag = l.rsplit('/',1)
                        self.word_dict[word] = self.word_dict.get(word, 0) + 1
                        self.tag_dict[tag] = self.tag_dict.get(tag, 0) + 1
    '''
    Loads a dataset. Specifically, returns a list of sentence_ids, and
    dictionaries of tag_lists and word_lists such that:
    tag_lists[sentence_id] = list of part-of-speech tags in the sentence
    word_lists[sentence_id] = list of words in the sentence
    ''' 
    def load_data(self, data_set):
        sentence_ids = []
        tag_lists = dict()
        word_lists = dict()
        sen_count = 0
    
        # iterate over documents
        for root, dirs, files in os.walk(data_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    for line in f:
                        pairs = line.strip().split() #split and strip from each line
                        
                        #Get words and tags
                        words_tags = [pair.rsplit('/', 1) for pair in pairs if '/' in pair]
                        words, tags = zip(*words_tags) if words_tags else ([], [])

                        # put the gotten words and tags into word_lists and tag_lists
                        tag_lists[sen_count] = tags #and here
                        word_lists[sen_count] = words #changed from list()
                        sentence_ids.append(sen_count)
                        sen_count += 1
    
        return sentence_ids, tag_lists, word_lists


    '''
    Implements the Viterbi algorithm.
    Use v and backpointer to find the best_path.
    '''
    def viterbi(self, sentence):
        T = len(sentence)
        N = len(self.tag_dict)
        v = np.zeros((N, T))
        backpointer = np.zeros((N, T), dtype=int)
    
        if T > 0:
            backpointer[:, 0] = 0 #Get backpointer and v
            v[:, 0] = self.initial + self.emission[sentence[0]]
    
            # recursion step
            for t in range(1, T):
                vt = v[:, t - 1, None] + self.transition + self.emission[sentence[t]]
                v[:, t] = np.max(vt, axis=0)
                backpointer[:, t] = np.argmax(vt, axis=0)
    
            b_point = np.argmax(v[:, T - 1], axis=0)
            best_path = [b_point]
            
            
            for i in range(T - 1, 0, -1):
                b_point = backpointer[b_point, i]
                best_path.insert(0, b_point)
    
            best_path = np.array(best_path)
        else:
            best_path = np.array([])
    
        return best_path #changed from .tolist()



    '''
    Trains a structured perceptron part-of-speech tagger on a training set.
    '''
    def train(self, train_set):
        self.make_dicts(train_set)
        sentence_ids, tag_lists, word_lists = self.load_data(train_set)
        Random(0).shuffle(sentence_ids)
        self.initial = np.zeros(len(self.tag_dict))
        self.transition = np.zeros((len(self.tag_dict), len(self.tag_dict)))
        self.emission = np.zeros((len(self.word_dict), len(self.tag_dict)))
        for i, sentence_id in enumerate(sentence_ids):
            sentence_words = word_lists[sentence_id]
            sentence_tags = tag_lists[sentence_id]
            
            # Get the words and put into sentence_words
            sentence_words = [list(self.word_dict.keys()).index(w) if w in self.word_dict.keys() else self.unk_index for w in sentence_words]
            
            # Get the tags and put into sentence_tags
            sentence_tags = [list(self.tag_dict.keys()).index(t) if t in self.tag_dict.keys() else self.unk_index for t in sentence_tags]
            
            predicted_tags = self.viterbi(sentence_words)
            
            if len(predicted_tags) > 0:
                # Add to initial, transition, and emission
                for x in range(0,len(sentence_words)):
                    if x <= 0 and int(predicted_tags[x]) == sentence_tags[x]:
                        self.initial[sentence_tags[x]] += 1
                        self.initial[predicted_tags[x]] -= 1
                        self.emission[sentence_words[x]][sentence_tags[x]] += 1
                        self.emission[sentence_words[x]][predicted_tags[x]] -= 1
                    else:
                        self.transition[sentence_tags[x]-1][sentence_tags[x]] += 1
                        self.transition[predicted_tags[x]-1][predicted_tags[x]] -= 1
                        self.emission[sentence_words[x]][sentence_tags[x]] += 1
                        self.emission[sentence_words[x]][predicted_tags[x]] -= 1
                        
            if (i + 1) % 1000 == 0 or i + 1 == len(sentence_ids):
                print(i + 1, 'training sentences tagged')






    '''
    Tests the tagger on a development or test set.
    Returns a dictionary of sentence_ids mapped to their correct and predicted
    sequences of part-of-speech tags such that:
    results[sentence_id]['correct'] = correct sequence of tags
    results[sentence_id]['predicted'] = predicted sequence of tags
    '''
    
    def test(self, dev_set):
        results = defaultdict(dict)
        sentence_ids, tag_lists, word_lists = self.load_data(dev_set)
    
        for i, sentence_id in enumerate(sentence_ids):
            word_test = [list(self.word_dict.keys()).index(word) if word in self.word_dict else self.unk_index for word in word_lists[sentence_id]]
            tag_test = [list(self.tag_dict.keys()).index(tag) if tag in self.tag_dict else self.unk_index for tag in tag_lists[sentence_id]]
    
            results[sentence_id]['correct'] = tag_test
            pred_test = self.viterbi(word_test)
            results[sentence_id]['predicted'] = pred_test
    
            if (i + 1) % 1000 == 0 or i + 1 == len(sentence_ids):
                print(i + 1, 'testing sentences tagged')
    
        return results




    '''
    Given results, calculates overall accuracy.
    '''
    def evaluate(self, results):
        correct_count = 0
        total_count = 0
    
        for result in results.values():
            for correct,predict in zip(result['correct'],result['predicted']):
                total_count+=1
                if correct == predict:
                    correct_count+=1
    
        accuracy = correct_count / total_count * 100
        return accuracy



if __name__ == '__main__':
    pos = POSTagger()
    # make sure these point to the right directories
    pos.train('brown/train')
    #pos.train('data_small/train')
    results = pos.test('brown/dev')
    #results = pos.test('data_small/test')
    print('Accuracy:', pos.evaluate(results))

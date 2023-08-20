# CS114B Spring 2023 Homework 1
# Steven Rud
# Naive Bayes in Numpy

import os
import numpy as np
from collections import defaultdict

class NaiveBayes():

    def __init__(self):
        self.class_dict = {}
        self.feature_dict = {}
        self.prior = None
        self.likelihood = None
        self.temp = []

    '''
    Trains a multinomial Naive Bayes classifier on a training set.
    Specifically, fills in self.prior and self.likelihood such that:
    self.prior[class] = log(P(class))
    self.likelihood[feature, class] = log(P(feature|class))
    '''
    def train(self, train_set):
        pos_dict = {}
        neg_dict = {}
        total_count = 0
        pos_count = 0
        neg_count = 0
        self.class_dict['neg'] = 0
        self.class_dict['pos'] = 1
        self.prior = np.zeros(len(self.class_dict))
        # iterate over training documents
        for root, dirs, files in os.walk(train_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    if 'neg' in root: #go through the neg and then pos folder and add values to a dict
                        for sent in f:
                            for w in sent.split():
                                neg_dict[w] = 1 + neg_dict.get(w, 0)
                        neg_count += 1
                        total_count += 1
                    elif 'pos' in root:
                        for sent in f:
                            for w in sent.split():
                                pos_dict[w] = 1 + pos_dict.get(w, 0)
                        pos_count +=1
                        total_count +=1

        total_count = neg_count + pos_count
        # calculate prior for each class

        self.prior[0] = np.log(neg_count/total_count) #set up prior
        self.prior[1] = np.log(pos_count / total_count)
        self.feature_dict = neg_dict.copy() #set up feature_dict with negative values here and positive in for loop
        for p in pos_dict:
            if p not in self.feature_dict:
                self.feature_dict[p] = 0
        self.feature_dict = {fe: temp_c for temp_c, fe in enumerate(self.feature_dict)}

        # calculate likelihood for each feature and class
        num_features = len(self.feature_dict)
        num_classes = len(self.class_dict)
        self.likelihood = np.zeros((num_features, num_classes)) #get shape of likelihood
        for x, feat in enumerate(self.feature_dict):
            for y in range(num_classes):
                comb_dict = pos_dict if y == 1 else neg_dict #get count and dict to add into likelihood
                d_count = comb_dict.get(feat, 0) + 1
                self.likelihood[x, y] = np.log(d_count / (sum(comb_dict.values()) + len(self.feature_dict)))

    '''
    Tests the classifier on a development or test set.
    Returns a dictionary of filenames mapped to their correct and predicted
    classes such that:
    results[filename]['correct'] = correct class
    results[filename]['predicted'] = predicted class
    '''
    def test(self, dev_set):
        results = defaultdict(dict)
        # iterate over testing documents
        for root, dirs, files in os.walk(dev_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    # create feature vectors for each document
                    temp_dict = dict()
                    for sent in f:
                        for w in sent.split():
                            temp_dict[w] = temp_dict.get(w, 0) + 1 #store document data in temp dictionary
                    feature_vector = {feature: temp_dict.get(feature, 0) for feature in self.feature_dict} #dictionary comprehension for adding features into a feature vector
                    class_k = list(self.class_dict.keys())

                    test_arg = np.argmax(np.add(self.prior, np.dot(list(feature_vector.values()),self.likelihood))) #the arg max using np.dot with prior and likelihood

                    results[name]['correct'] = os.path.basename(root) #name of folder
                    results[name]['predicted'] = class_k[test_arg] #gets predicted
        return results

    '''
    Given results, calculates the following:
    Precision, Recall, F1 for each class
    Accuracy overall
    Also, prints evaluation metrics in readable format.
    '''
    def evaluate(self, results):
        # you may find this helpful
        confusion_matrix = np.zeros((len(self.class_dict),len(self.class_dict)))
        true_pos = defaultdict(int) #get all false and trues
        false_pos = defaultdict(int)
        false_neg = defaultdict(int)
        for filename, r in results.items():
            correct = r['correct']
            predicted = r['predicted']
            confusion_matrix[self.class_dict[correct], self.class_dict[predicted]] += 1
            if correct != predicted: #if correct and predict don't match then add to false
                false_pos[predicted] += 1
                false_neg[correct] += 1
            else: #else add to true
                true_pos[correct] += 1
        print('Confusion Matrix:')
        print(confusion_matrix)
        for class_name in self.class_dict.keys(): #using name of classing and previously calcuted trues and false get precision recall and f1
            precision = true_pos[class_name] / (true_pos[class_name] + false_pos[class_name])
            recall = true_pos[class_name] / (true_pos[class_name] + false_neg[class_name])
            f1 = 2 * precision * recall / (precision + recall)
            print('For the', class_name,'class') #Printing
            print('Precision:', precision)
            print('Recall:', recall)
            print('F1:', f1)
            print()
        accuracy = sum(true_pos.values()) / sum(confusion_matrix.flatten()) #sums everything that is needed and gets accuracy
        print('Accuracy:', accuracy)

if __name__ == '__main__':
    nb = NaiveBayes()
    # make sure these point to the right directories
    nb.train('movie_reviews/train')
    #nb.train('movie_reviews_small/train')
    results = nb.test('movie_reviews/dev')
    #results = nb.test('movie_reviews_small/test')
    nb.evaluate(results)

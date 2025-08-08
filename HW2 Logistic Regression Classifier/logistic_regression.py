# CS114B Spring 2023 Homework 2
# Logistic Regression Classifier
# Steven Rud

import os
import numpy as np
from collections import defaultdict
from math import ceil
from random import Random
from scipy.special import expit # logistic (sigmoid) function

class LogisticRegression():

    def __init__(self):
        self.class_dict = {}
        # use of self.feature_dict is optional for this assignment
        self.feature_dict = {}
        self.n_features = None
        self.theta = None # weights (and bias)

    '''
    Given a training set, fills in self.class_dict (and optionally,
    self.feature_dict), as in HW1.
    Also sets the number of features self.n_features and initializes the
    parameter vector self.theta.
    '''
    def make_dicts(self, train_set):
        # iterate over training documents
        for root, dirs, files in os.walk(train_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    # get the class name from the file path
                    class_name = os.path.basename(os.path.normpath(root))
                    if class_name not in self.class_dict:
                        self.class_dict[class_name] = len(self.class_dict) #fills in class dict
        # rest of variables being filled
        self.feature_dict = {'fast': 0, 'couple': 1, 'shoot': 2, 'fly': 3, 'run':4, 'funny': 5, 'car':6, 'chase':7, 'movie': 8, 'boring': 9}
       # print(self.feature_dict)
        self.n_features = 4
        self.theta = np.zeros(self.n_features + 1)

    '''
    Loads a dataset. Specifically, returns a list of filenames, and dictionaries
    of classes and documents such that:
    classes[filename] = class of the document
    documents[filename] = feature vector for the document (use self.featurize)
    '''
    def load_data(self, data_set):
        # Initialize variables
        filenames =[]
        classes = {}
        documents = {}
        end_root = len(list(self.class_dict.keys())[0])
        #print(end_root)
        for root, dirs, files in os.walk(data_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    word_type = {}
                    if name.endswith('.txt'):
                        filenames.append(name)
                        classes[name] = root[-end_root:] # gets the pos or neg value from path
                        words = f.read().split()
                        wt = [word for word in words]
                        word_type = {word: wt.count(word) for word in wt} #fills up all the words
                        documents[name] = self.featurize(word_type) #featurizes the words and adds to document
        
        return filenames, classes, documents

    
    '''
    Given a document (as a list of words), returns a feature vector.
    Note that the last element of the vector, corresponding to the bias, is a
    "dummy feature" with value 1.
    '''
    def featurize(self, document):
        vector = np.zeros(self.n_features + 1, dtype=int)
        keys = list(self.feature_dict.keys())
        word_start = sum(1 for word in document if word in keys[:len(keys) // 2]) #get start and end for words in documents
        word_end = sum(1 for word in document if word in keys[len(keys) // 2:])
        vector[0] = len(document) / 100000 #edit first vector, middle vectors, and final vector
        vector[1:-1:2] = word_start
        vector[2:-1:2] = word_end
        vector[-1] = 1
        #print(vector)
        return vector

    '''
    Trains a logistic regression classifier on a training set.
    '''
    def train(self, train_set, batch_size=16, n_epochs=5, eta=0.06):
        # Load data
        filenames, classes, documents = self.load_data(train_set)
        filenames = sorted(filenames)
    
        # Compute number of minibatches
        n_minibatches = ceil(len(filenames) / batch_size)
    
        # Initialize loss
        loss = 0
    
        # Train model
        for epoch in range(n_epochs):
            print("Epoch {:} out of {:}".format(epoch + 1, n_epochs))
    
            # Iterate over minibatches
            for i in range(n_minibatches):
                # Get minibatch of filenames
                minibatch = filenames[i * batch_size: (i + 1) * batch_size]
    
                # Create the matrix x and the vector y
                x = np.ones((len(minibatch), self.n_features + 1), dtype = int)   
                y = np.empty((len(minibatch), ), dtype = int)
    
                # Fill in matrix x and vector y
                for j, feat_name in enumerate(minibatch):
                    if feat_name in documents:
                        x[j] = documents[feat_name]
    
                    if feat_name in classes:
                        if classes[feat_name][-3:] == 'pos': #if a pos review give a value of 1 else 0
                            y[j] = 1
                        else:
                            y[j] = 0
    
                # Compute y_hat and update loss
                y_hat = expit(np.dot(x, self.theta))
                loss += -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    
                # Compute gradients and update theta
                grad = 1/len(minibatch) * (x.transpose() @ (y_hat - y))
                self.theta = self.theta - (eta * grad) 
    
            # Get average loss
            loss /= len(filenames)
            print("Average Train Loss: {}".format(np.sum(loss)))
    
            # Randomize order
            Random(epoch).shuffle(filenames)


    '''
    Tests the classifier on a development or test set.
    Returns a dictionary of filenames mapped to their correct and predicted
    classes such that:
    results[filename]['correct'] = correct class
    results[filename]['predicted'] = predicted class
    '''     
    def test(self, dev_set):
        filenames, classes, documents = self.load_data(dev_set)
        results = defaultdict(dict)
        for name in filenames:
            y_hat = expit(np.dot(documents[name], self.theta)) #get y_hat to find predicted value and then add correct value from data.
            predicted_class = list(self.class_dict.keys())[1] if y_hat > 0.5 else list(self.class_dict.keys())[0]
            results[name]['predicted'] = predicted_class
            results[name]['correct'] = classes[name]
    
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
    lr = LogisticRegression()
    # make sure these point to the right directories
    lr.make_dicts('movie_reviews/train')
    #lr.make_dicts('movie_reviews_small/train')
    lr.train('movie_reviews/train', batch_size=16, n_epochs=5, eta=0.06)
    #lr.train('movie_reviews_small/train', batch_size=3, n_epochs=1, eta=0.1)
    results = lr.test('movie_reviews/dev')
    #results = lr.test('movie_reviews_small/test')
    lr.evaluate(results)

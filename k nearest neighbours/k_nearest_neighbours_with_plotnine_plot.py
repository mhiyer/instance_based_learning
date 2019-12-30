# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 16:38:18 2019

@author: mh iyer

Development of k-nearest neighbours using Python3
The target variable does not need to be numeric- it can be a string. 
e.g. 'Sunny', 'Versicolor' etc. are allowed.

A sample plot depicting the classification result is also created 
using a python version of ggplot - plotnine library

Specifications:
    To create model:
        Inputs:
            training data- in the form of a pandas dataframe
            k - number of neighbours
            target column name- the column name in the pandas dataframe that contains the targets
            e.g. 'ground_truth', 'species' etc. 
        Output: kNN Model
    
    To get predictions:
        Input:
            testing data - in the form of a pandas dataframe
        Output: 
            predictions- these are class predictions- whether they are numeric or strings 
                        is dependent on your target column definition
            accuracy- the percentage of correctly classified test examples
                        updated test data pandas dataframe- this has an extra column which specifies 
                        whether a given example is correctly classified or misclassified.
                        This could be useful in plotting or just as a record of sorts
        
"""

 
import numpy as np
import pandas as pd
import math
from collections import Counter
import random
from plotnine import *
# turn off the 'SettingWithCopyWarning' (optional)
pd.set_option('mode.chained_assignment', None)

random.seed(1234)

# function to shuffle pandas dataframe
def shuffle_data(dataset):
    # create extra column which is basically the index shuffled
    dataset['shuffled_index'] = random.sample(range(0,len(dataset)), len(dataset))
    
    # sort this
    dataset = dataset.sort_values(by=['shuffled_index'])
    
    # set the new column (now sorted)
    dataset = dataset.set_index(['shuffled_index'])
    
    return dataset


# define k-nearest neighbour class
class kNN:
    # initialize
    def __init__(self, dataset, k, target_column_name):
        # training data here
        self.dataset = dataset
        
        # user-defined number of neighbours
        self.k = k
        
        # get target column name
        self.target_column = target_column_name
        
        # get all the column names, save the target column
        self.feature_columns = [x for x in self.dataset.columns if x != self.target_column]
        
        # get the number of possible targets- they DON'T HAVE TO BE NUMBERS
        # since we are looking at the most common target, they could be strings!
        self.target_names = self.dataset[self.target_column].unique()
        
        # number of features
        self.number_of_features = len(self.dataset.columns)-1
        
        # process
        self.X, self.y = self.convert_data(self.dataset)
        
        # training over! the data 'sits there' and waits till a new example is presented to it, then it does its magic
    
    
    # extract two sets of data - the first corresponds to the features, the second to the target
    def convert_data(self, data):
        
        # get a list of lists of features (without target), and a list for targets
        # initialize list for X
        X = []
        # initialize list for y(target)
        y = []
        
        # loop through to get lists
        for i in range(0,len(data)):
            x = data.loc[i]
            # the reason we're doing this is because the test dataset may have a different ordering of columns, for whatever reason
            features_x = [x[val] for val in self.feature_columns]
            
            X.append(features_x)
            y.append(x[self.target_column])

        return(X,y)        

    # get euclidean distance between two vectors
    def euclidean_distance(self, vector1, vector2):
        dist = 0
        for i in range(0, self.number_of_features):
            dist+=(vector1[i]-vector2[i])**2
        dist_sqrt = math.sqrt(dist)
        return(dist_sqrt)
        
    # get k nearest neighbours by computing distances for a point across all the points in the training space
    def get_neighbours(self):
        # get a list of distances from the point to all the training examples
        self.distances = []
        # loop through the training examples
        for training_example in self.X:
            self.distances.append(self.euclidean_distance(self.current_test_pt, training_example))    
        
        # get the distances sorted, obtain the sorted indices, then get the targets of the neighbours
        # get the first k sorted indices ONLY
        sorted_indices = np.argsort(self.distances)[:self.k]
        self.neighbours = [self.X[x] for x in sorted_indices]
        self.target_values_of_neighbours = Counter([self.y[x] for x in sorted_indices])
        
    # get a vote of the neighbours as to what value to assign to the test point being analyzed
    # in effect, get the most common target shared
    def vote(self):
        # define a value for max_occurrence and class assigned
        max_occurrence = 0
        self.class_assigned = []
        
        # loop through the target names
        for target_name in self.target_values_of_neighbours.keys():
            if self.target_values_of_neighbours[target_name]>max_occurrence:
                max_occurrence = self.target_values_of_neighbours[target_name]
                self.class_assigned = target_name
        
        #print('class assigned is:',self.class_assigned,'with ',max_occurrence,' votes!')
            
                        
    # predict on new instances - they are supposed to be new, in the sense they are not in the training set
    # but for analysis, of course they could be, but kindly be mindful that 
    # prediction on training examples is NOT necessarily representative of prediction on new examples
    def predict(self, test_data):
        
        # define empty list for predictions
        predictions = []
        
        # convert test data to the format used for training
        test_X, test_y = self.convert_data(test_data)
        
        # for each example, get the 'k' nearest neighbours by computing distances (L2) across 
        # all the training examples
        
        # loop through the test points
        for test_pt in test_X:
            self.current_test_pt = test_pt
            
            # get neighbours
            self.get_neighbours()
            
            # return the most common target value of the neighbours- in effect, take a vote!
            self.vote()
            
            # append to predictions
            predictions.append(self.class_assigned)
        
        
        # add a new column which takes on values 'correctly_classified', and 'misclassified'
        # first initialize a list
        classification_status = []
        
        # compute accuracy and misclassified predictions
        correct_count = 0 
        for i in range(0,len(predictions)):
            if predictions[i]==test_y[i]:
                correct_count+=1
                classification_status.append('correctly_classified')
            else:
                classification_status.append('misclassified')
        
        accuracy = correct_count/len(predictions)
        test_data['Classification']=classification_status
        return(predictions, accuracy, test_data)
       

            
# usage of this function in practice
if __name__ == "__main__":
    
    # open csv
    path = r'iris.csv'
    data = pd.read_csv(path)
    
    # shuffle data
    shuffled_data = shuffle_data(data)
        
    # train 
    train_data = shuffled_data[0:100]
    
    # test
    test_data = shuffled_data[100:len(data)]
    # reset index to start from 0
    test_data.index = range(0,len(test_data))
    
    # define some parameters
    k = 5
    target_column_name = 'species'
    
    # initialize class 
    knn= kNN(train_data, k, target_column_name)

    # get predictions
    predictions, accuracy, test_data_with_classification_status = knn.predict(test_data)

    # print the accuracy
    print('the accuracy is :',accuracy)

    # visualize data - you will need to change some parameters depending on your data( column names could differ)
    g = ggplot(test_data_with_classification_status, aes(y = 'sepal_length', x = 'sepal_width', color = 'Classification')) +    geom_point(shape='o',fill='none', stroke=1,size=4) 
    g = g + ggtitle("Misclassifications with the Model") 
    print(g)        

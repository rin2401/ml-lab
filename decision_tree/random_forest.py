import numpy as np 
import pandas as pd
from graphviz import *
from decision_tree import *

class RandomForest(object):
    def __init__(self, num_tree, random_attribute = False):
        self.forest = []
        self.num_tree = num_tree
        self.random_attribute = random_attribute

    def fit(self, data, target):
        num_data = data.shape[0]
        num_attr = data.count(1).iloc[0]
        for i in range(self.num_tree):
            idx = np.random.choice(num_data, size=num_data//2, replace=False, p=None)
            tree = DecisionTree()

            if self.random_attribute:
                col_idx = np.random.choice(num_attr, size=9*num_attr//10, replace=False, p=None)
                col_idx.sort()
                tree.fit(data.iloc[idx,col_idx], target.iloc[idx])
            else:
                tree.fit(data.iloc[idx], target.iloc[idx])

            self.forest.append(tree)

    def draw_forest(self, filename="tree"):
        for i, tree in enumerate(self.forest):
           tree.draw_tree(filename+"_"+str(i))         

    def predict(self, data):
        votes = []
        for tree in self.forest:
            votes.append(tree.predict(data))

        results = [None]*len(data)
        for i in range(len(data)):
            lst = [row[i] for row in votes]
            results[i] = max(lst, key=lst.count)
        return results
    
    def score(self, data, target):
        labels = self.predict(data)
        return sum(labels==target)/len(target)
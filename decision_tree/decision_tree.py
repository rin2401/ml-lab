import numpy as np 
import pandas as pd
from graphviz import *

class Node(object):
    def __init__(self, example_indexs=None , attribute_indexs=None, entropy=0):
        self.example_indexs = example_indexs
        self.attribute_indexs = attribute_indexs
        self.attribute_index = None
        self.entropy = entropy
        self.attribute_values = None
        self.children = []
        self.label = None
        self.index = None

class DecisionTree(object):
    def __init__(self):
        self.root = None
        self.num_node=0

    def fit(self, data, target):
        self.target = target 
        self.num_train = len(data)
        self.attributes = list(data)
        self.labels = target.unique()
        example_indexs = range(self.num_train)
        data.index = example_indexs
        self.data = data 
        attribute_indexs = list(range(len(self.attributes)))
        self.root = Node(example_indexs, attribute_indexs, self.entropy(example_indexs))
        self.choose_attribute(self.root)
        
    def choose_attribute(self, node):
        self.num_node += 1
        node.index=self.num_node
        labels = self.target.iloc[node.example_indexs].unique()
        if len(labels) == 1:
            node.label = labels[0]
            return
        
        if len(node.attribute_indexs)==0:
            node.label = self.target.iloc[node.example_indexs].mode()[0]
            return

        max_gain = 0
        best_splits = []
        for attribute_index in node.attribute_indexs:
            sub_data = self.data.iloc[node.example_indexs, attribute_index]
            attribute_values = sub_data.unique().tolist()

            remainder = 0
            splits = []
            for value in attribute_values:
                split_indexs = sub_data.index[sub_data==value].tolist()
                remainder += len(split_indexs)/len(sub_data)*self.entropy(split_indexs)
                splits.append(split_indexs)
            
            gain = node.entropy - remainder
            if gain > max_gain:
                max_gain = gain
                best_splits = splits
                node.attribute_index = attribute_index
                node.attribute_values = attribute_values
        
        remain_attributes = node.attribute_indexs
        if node.attribute_index in remain_attributes:
            remain_attributes.remove(node.attribute_index)
        for split in best_splits:
            child_node = Node(split, remain_attributes, self.entropy(split))
            node.children.append(child_node)
            self.choose_attribute(child_node)

    def entropy(self, example_indexs):
        if len(example_indexs) == 0:
            return 0
        freqs = np.array(self.target.iloc[example_indexs].value_counts())
        freqs = freqs[freqs>0]
        probs = freqs / len(example_indexs)
        return -np.sum(probs*np.log2(probs))

    def draw_tree(self, filename="graph"):
        g = Digraph()
        stack = [self.root]

        while stack:
            node = stack.pop()
            start_index = str(node.index)
            if node.label:
                start_name = str(node.label)
            else:
                start_name = self.attributes[node.attribute_index]

            g.node(start_index, start_name)
            for i,child in enumerate(node.children):
                label = node.attribute_values[i]
                end_index = str(child.index)
                g.edge(start_index, end_index, label)                
                stack.append(child)
        
        g.render(filename=filename, directory="image",  format="png")
        return g
    
    def predict(self, data):
        num_data = len(data)
        labels = [None]*num_data
        for i in range(num_data):
            x = data.iloc[i, :]
            node = self.root
            while node.children:
                value = x.iloc[node.attribute_index]
                if value not in node.attribute_values:
                    break
                value_index = node.attribute_values.index(value)
                node = node.children[value_index]
            if node.label:
                labels[i] = node.label
            
        return labels

    def score(self, data, target):
        labels = self.predict(data)
        return sum(labels==target)/len(target)
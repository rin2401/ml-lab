import numpy as np 
import pandas as pd
from graphviz import *
from decision_tree import *
from random_forest import *

if __name__ == "__main__":
    # df = pd.read_csv("data/weather.csv")
    # data = df.iloc[:,1:-1]
    # target = df.iloc[:, -1]

    df = pd.read_csv("data/restaurant.csv")
    data = df.iloc[:,1:-1]
    target = df.iloc[:, -1]

    # df = pd.read_csv("data/mushrooms.csv")
    # data = df.iloc[:,1:-1]
    # target = df.iloc[:, 0]
    

    # n_train = int(len(data)*0.7)
    # X_train = data.iloc[:n_train]
    # X_test = data.iloc[n_train:]
    # y_train = target.iloc[:n_train]
    # y_test = target.iloc[n_train:]


    tree = DecisionTree()
    tree.fit(data, target)
    tree.draw_tree()
    print(tree.predict(data))

    # score = tree.score(data, target)
    # print(score)

    # forest = RandomForest(7)
    # forest.fit(X_train, y_train)
    # forest.draw_forest()
    # # print(forest.predict(data))
    # print(forest.score(X_test, y_test))


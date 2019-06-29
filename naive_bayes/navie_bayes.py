import numpy as np
import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split


class NaiveBayes(object):

    def __init__(self):
        self.word = None
        self.count = None
        self.label = None
        self.prob_word_label = None
        self.prob_label = None

    def fit(self, X, y):
        self.word , self.count =  self.count_word(X)
        self.label, count_label = np.unique(y, return_counts=True)
        self.prob_label = count_label/np.sum(count_label)
        self.prob_word_label = self.caculate_prob(X, y)
    
    def count_word(self, X):
        w = set()
        for x in X:
            w.update(x.split())
        word = list(w)
        count = np.array([[0]*len(word)]*len(X))
        for i, x in enumerate(X):
            idxs = [word.index(j) for j in x.split()]
            count[i,idxs] += 1

        return word, count

    def caculate_prob(self, X, y):
        prob = np.array([[0.0]*len(self.label)]*len(self.word))
        for i, l in enumerate(self.label):
            idx = [k for k,j in enumerate(y) if j==l]
            word_by_class = np.sum(self.count[idx], axis=0)
            prob[:,i] = (word_by_class+1)/(np.sum(word_by_class) + len(word_by_class))

        return prob

    def predict(self, X):
        preds = np.array([np.log(self.prob_label)]*X.shape[0])
        for i, x in enumerate(X):
            idxs = [self.word.index(j) for j in x.split() if j in self.word]
            x_prob = self.prob_word_label[idxs,:]
            log_probs = np.log(x_prob)
            preds[i,:] += np.sum(log_probs, axis=0)

        return self.label[np.argmax(preds, axis=1)]       
 
    def score(self, X, y):
        preds = self.predict(X)
        accuracy = np.sum(preds==list(y))/len(X)
        
        return accuracy



news_df = pd.read_csv("uci-news-aggregator.csv", sep = ",")
news_df = news_df[:10000]

print(news_df.CATEGORY.unique())

news_df['CATEGORY'] = news_df.CATEGORY.map({ 'b': 1, 't': 2, 'e': 3, 'm': 4 })
news_df['TITLE'] = news_df.TITLE.map(
    lambda x: x.lower().translate(str.maketrans('','', string.punctuation))
)


X_train, X_test, y_train, y_test = train_test_split(
    news_df['TITLE'], 
    news_df['CATEGORY'], 
    random_state = 1
)

print("Training dataset: ", X_train.shape[0])
print("Test dataset: ", X_test.shape[0])

clf = NaiveBayes()
clf.fit(X_train, y_train)
print("Accuracy score", clf.score(X_test, y_test))


## sklearn
print("Using sklearn")
count_vector = CountVectorizer(stop_words = 'english')
training_data = count_vector.fit_transform(X_train)
testing_data = count_vector.transform(X_test)


naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)
predictions = naive_bayes.predict(testing_data)

print("Accuracy score: ", accuracy_score(y_test, predictions))
print("Recall score: ", recall_score(y_test, predictions, average = 'weighted'))
print("Precision score: ", precision_score(y_test, predictions, average = 'weighted'))
print("F1 score: ", f1_score(y_test, predictions, average = 'weighted'))
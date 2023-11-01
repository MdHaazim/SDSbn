#import needed libraries
import os
import numpy as np
from collections import Counter
import pandas as pd
from nltk.corpus import stopwords      # contains list of stopwords

#create word dictionary
def make_dictionary(train_dir):
    # ref https://www.kdnuggets.com/2020/07/spam-filter-python-naive-bayes-scratch.html
    
    train_set = pd.read_csv(train_dir)
    train_set['email'] = train_set['email'].str.split()
    vocab = []
    for mail in train_set['email']:
        try:
            for word in mail:
                vocab.append(str(word))
        except TypeError:   # Data Set used has multiple errors when ran, so this is to catch it
            #print("cant")
            continue
    
    dictionary = Counter(vocab)
    #print(len(vocab))       #print number of unique words in train set
    
    list_to_remove = dictionary.keys()
    for item in list(list_to_remove):
        if item.isalpha() == False:
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
        elif item in stopwords.words('english'):    # removes stop words
            del dictionary[item]
            
    dictionary = dictionary.most_common(3000)
    #print(dictionary)
    return dictionary
    
#feature extraction

#datasets
train_dir = "spam_or_not_spam.csv"

#train classifiers

#print results
print(make_dictionary(train_dir))
#import needed libraries
import os
import numpy as np
from collections import Counter
import pandas as pd

#create word dictionary
def make_dictionary(train_dir):
    words = pd.read_csv(train_dir).to_string()
    dictionary = Counter(words.split())
    
    list_to_remove = dictionary.keys()
    for item in list(list_to_remove):
        if item.isalpha() == False:
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(3000)
    print(dictionary)
    return dictionary
    # print(emails[0])
    # for mail in emails:
    #     with open(mail) as m:
    #         for i, line in enumerate(m):
    #             if i == 2
#feature extraction

#datasets
train_dir = "spam_or_not_spam.csv"

#train classifiers

#print results
make_dictionary(train_dir)
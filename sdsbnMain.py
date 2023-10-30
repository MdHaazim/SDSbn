#import needed libraries
import os
import numpy as np
from collections import Counter

#create word dictionary
def make_dictionary(train_dir):
    emails = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    allWords = []

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
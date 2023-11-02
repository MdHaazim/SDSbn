#import needed libraries
import os
import numpy as np
from collections import Counter
import pandas as pd
from nltk.corpus import stopwords      # contains list of stopwords
from sklearn.svm import LinearSVC #SVC, NuSVC, 
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB #BernoulliNB,

#create word dictionary
def make_dictionary(train_dir):
    # ref https://www.kdnuggets.com/2020/07/spam-filter-python-naive-bayes-scratch.html
    
    train_set = pd.read_csv(train_dir)        # Loads folder instead of using OS since it is a .csv file
    train_set['email'] = train_set['email'].str.split()        # Takes only the email column and splits into individual words
    vocab = []        # Stores all the words in data set
    for mail in train_set['email']:
        try:
            for word in mail:
                vocab.append(str(word))
        except TypeError:   # Data Set used has multiple errors when ran, so this is to catch it
            #print("cant")
            continue
    
    dictionary = Counter(vocab)    # From here it follows the slides basically
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
def read_email_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def preprocess_text(text):
    # Tokenization (split the text into words)
    words = nltk.word_tokenize(text)
    
    # Convert to lowercase and remove non-alphabetic characters
    words = [word.lower() for word in words if word.isalpha()]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Join the words back into a single string
    processed_text = ' '.join(words)
    
    return processed_text
def extract_features(text, method='count'):
    if method == 'count':
        vectorizer = CountVectorizer()
    else:
        raise ValueError("Invalid feature extraction method. Use 'count'")

    feature_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()

    return feature_matrix, feature_names
    
def save_features_to_csv(features, feature_names, output_csv):
    df = pd.DataFrame(features.toarray(), columns=feature_names)
    df.to_csv(output_csv, index=False)
     
#datasets
train_dir = "spam_or_not_spam.csv"

#train classifiers
#Training SVM and Naive Bayes Classifier

model1=MultinomialNB()
model2=LinearSVC()
model1.fit(train_matrix,train_label)
model2.fit(train_matrix,train_label)

#Test the unseen mails for Spam
#test_dir= 'test-mails'
#test_matrix= extract_features(test_dir)
#test_labels=np.zeros(260)
#test_labels[130:260]=1

#result1=model1.predict(test_matrix)
#result2=model2.predict(test_matrix)
#matriz=confusion_matrix(test_labels,result1)
#matric=confusion_matrix(test_labels, result2)   #Not sure abt test mails

#print results
print(make_dictionary(train_dir))

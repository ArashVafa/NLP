import csv

reviewTexts_train = []
sentiment_train = []

reviewTexts_test = []
sentiment_test = []

with open('reviews_using_dataset.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',') 
    for i, row in enumerate(readCSV):
        if i==0:
            continue
        if i < 100:
            reviewTexts_train.append(row[0])
            sentiment_train.append(int(row[1]))
        else:
            break
            reviewTexts_test.append(row[0])
            sentiment_test.append(int(row[1]))
            
from gensim.corpora import Dictionary
dct = Dictionary([sent.strip().split() for sent in reviewTexts_train])

doc_idx = [dct.doc2idx(reviewTexts_train[i].strip().split()) for i in range(len(reviewTexts_train))]
print (doc_idx[0])
#train_bow = dct.doc2bow([sent.strip().split() for sent in reviewTexts_train])

#print (len(train_bow))

from keras.preprocessing import sequence
max_words = 500
X_train = sequence.pad_sequences(doc_idx, maxlen=max_words)
#X_test = sequence.pad_sequences(X_test, maxlen=max_words)

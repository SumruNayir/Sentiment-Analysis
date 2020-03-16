from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import pandas as pd
import networkx as nx
from sklearn import preprocessing 
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding, CuDNNGRU
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

#colon name's polarity
datasetOrijinal = pd.read_csv("...csv")

dataset = pd.read_csv("....csv")
dataset['polarity'].unique() 

#Label Encoding
label_encoder = preprocessing.LabelEncoder() 
dataset['polarity']= label_encoder.fit_transform(dataset['polarity']) 
dataset['polarity'].unique() 

target = dataset['polarity'].values.tolist()
data = dataset['text'].values.tolist()

#0.80 data
cutoff = int(len(data) * 0.80)
x_train, x_test = data[:cutoff], data[cutoff:] #text
y_train, y_test = target[:cutoff], target[cutoff:] #etiket
#Control
#print(x_train[100])
#print(y_train[100])

num_words = 1000
tokenizer = Tokenizer(num_words = num_words)
tokenizer.fit_on_texts(data)
#print(a)


x_train_tokens = tokenizer.texts_to_sequences(x_train)
x_test_tokens = tokenizer.texts_to_sequences(x_test)
#Kontrol
print(x_train[100])
print(x_train_tokens[100])

#Size Synchronization
num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens]
num_tokens = np.array(num_tokens)

average = np.mean(num_tokens)
print("Average:", average)

counter = np.max(num_tokens)
print("Counter: ",counter)

index = np.argmax(num_tokens)
print("Index: ",index)


max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)
#print(max_tokens)
np.sum(num_tokens < max_tokens) / len(num_tokens)
 

x_train_pad =pad_sequences(x_train_tokens, maxlen=max_tokens)
x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens)
#Control
#print(x_train_pad.shape)
print(np.array(x_train_tokens[800])) 
print(x_train_pad[800]) 



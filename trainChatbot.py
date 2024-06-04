# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 22:42:25 2023

@author: gatik
"""
#Imports
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
import random

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

#Initialize
words=[]
classes=[]
documents=[]
ignore_words=['?','!','@','$']

#Using json
data_file= open('intents.json').read()
intents = json.loads(data_file)

#Populating the list
for intent in intents['intents']:
    for pattern in intent['patterns']:
        
        #Take each word and tokenize it
        w= nltk.word_tokenize(pattern)
        words.extend(w)
        
        #Adding documents
        documents.append((w,intent['tag']))
        
        #Adding classes to our class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

# print(len(documents)," Documents: ", documents)
# 
# print("\n")
# 
# print(len(classes), " Classes: ", classes)
# 
# print("\n")
# 
# print(len(words), " Unique Lemmatized Words: ", words)
# 
# print("\n")

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl','wb'))

#Initializing the training data
training = []
output_empty=[0]*len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
        
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])
print("training: ",training)

random.shuffle(training)
training = np.array(training)
train_x = list(training[:,0])
train_y = list(training[:,1])

print("Training Data Created")
print("train_x: ", train_x)
print("train_y: ", train_y)


model = Sequential()
model.add(Dense(128, input_shape = (len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))  
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print('Model created')       
        
        
        
        
        
        
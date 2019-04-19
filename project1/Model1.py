
###Model:1-level RNN

###
import gzip
import gensim 
import logging
import pandas as pd
import numpy as np
import jieba.analyse
import jieba
import codecs
import jieba.posseg as pseg
import warnings

from gensim.models import word2vec

import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

import gensim

#import scikitplot.plotters as skplt

import nltk

#from xgboost import XGBClassifier

import os

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam

def one_hot_encode(x,NumOfClass):
  
    encoded = np.zeros((len(x), NumOfClass))
    
    for i in range(len(x)):
        for j in range(NumOfClass):
            if(x[i]==j):
                encoded[i][j]=1

    
    return encoded



df = pd.read_csv('project1_data/train.csv')
#print(df.head())

# stopWords = []
# with open('project1_data/stopwords.txt', 'r',encoding='UTF-8') as file:
#     for data in file.readlines():
#         data = data.strip()
#         stopWords.append(data)

# stoplst = [' ', '\xa0']
# for words in stoplst:
#     stopWords.append(words)

# Text1=[]
# for i in range(len(df['title1_zh'])):
# 	try:
# 		poss = jieba.cut(df['title1_zh'][i], cut_all = False)
# 		Text1.append([])
# 		for w in poss:
# 			if w not in stopWords:
# 				Text1[-1].append(w) 
#                 #print("W:",w)       
#             # if names.get(w) is None and w not in stopWords:    
#             #     relationships[w] = {}            
# 	except:
# 		pass

# Text2=[]
# for i in range(len(df['title2_zh'])):
# 	try:
# 		poss2 = jieba.cut(df['title2_zh'][i], cut_all = False)
# 		Text2.append([])
# 		for w in poss2:
# 			if w not in stopWords:
# 				Text2[-1].append(w) 
#                 #print("W:",w)       
#             # if names.get(w) is None and w not in stopWords:    
#             #     relationships[w] = {}            
# 	except:
# 		pass

# Label=[]
# for i in range(len(df['label'])):
# 	try:
# 		if df['label'][i]=='agreed':
# 			Label.append(0)
# 		elif df['label'][i]=='disagreed':
# 			Label.append(1)
# 		else:
# 			Label.append(2)
# 	except:
# 		pass



# print("Text1",Text1[0])
# print("Text2",Text2[0])
# print("Label:",Label[0])



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


#X = ["Good morning", "Sweet Dreams", "Stay Awake"]
#Y = ["Good morning", "Sweet Dreams", "Stay Awake"]
#X = Text1
#Y = Label


Eng_Text=[]
for i in range(len(df['title1_en'])):
	try:
		
		Eng_Text.append([df['title1_en'][i]+" "+df['title2_en'][i]])
		 
	except:
		pass

print("text1:",Eng_Text[0])
print("text2:",Eng_Text[1])
#
Label=[]
for i in range(len(df['label'])):
	try:
		if df['label'][i]=='agreed':
			Label.append(0)
		elif df['label'][i]=='disagreed':
			Label.append(1)
		else:
			Label.append(2)
	except:
		pass


num_words = 200
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(Eng_Text)


X = tokenizer.texts_to_sequences(Eng_Text)
X = pad_sequences(X, maxlen=200)

embed_dim = 128
lstm_out = 196

# Model saving callback
ckpt_callback = ModelCheckpoint('keras_model', 
                                 monitor='val_loss', 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode='auto')

model = Sequential()
model.add(Embedding(num_words, embed_dim, input_length = X.shape[1]))
model.add(LSTM(lstm_out, recurrent_dropout=0.2, dropout=0.2))
model.add(Dense(3,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['categorical_crossentropy'])
print(model.summary())

Y = one_hot_encode(Label,3)

print(X.shape, Y.shape)

batch_size = 32
model.fit(X, Y, epochs=8, batch_size=batch_size, validation_split=0.2, callbacks=[ckpt_callback])




df_test = pd.read_csv('project1_data/test.csv')

Eng_Text_test=[]
for i in range(len(df_test['title1_en'])):
	try:
		
		Eng_Text_test.append([df_test['title1_en'][i]+" "+df['title2_en'][i]])
		 
	except:
		pass

print("text1:",Eng_Text_test[0])
print("text2:",Eng_Text_test[1])
#
Label_test=[]
for i in range(len(df_test['label'])):
	try:
		if df_test['label'][i]=='agreed':
			Label_test.append(0)
		elif df_test['label'][i]=='disagreed':
			Label_test.append(1)
		else:
			Label_test.append(2)
	except:
		pass


num_words = 200
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(Eng_Text_test)


X_test = tokenizer.texts_to_sequences(Eng_Text_test)
X_test = pad_sequences(X_test, maxlen=200)
Y_test = one_hot_encode(Label_test,3)


probas = model.predict(X_test)
pred_indices = np.argmax(probas, axis=1)
classes = np.array(range(1, 10))
preds = classes[pred_indices]
print('Log loss: {}'.format(log_loss(classes[np.argmax(Y_test, axis=1)], probas)))
print('Accuracy: {}'.format(accuracy_score(classes[np.argmax(Y_test, axis=1)], preds)))
skplt.plot_confusion_matrix(classes[np.argmax(Y_test, axis=1)], preds)



import gzip
import gensim 
import logging
import pandas as pd
import numpy as np
import jieba.analyse
import jieba
import codecs
import jieba.posseg as pseg


df = pd.read_csv('project1_data/train.csv')
print(df.head())

stopWords = []
with open('project1_data/stopwords.txt', 'r',encoding='UTF-8') as file:
    for data in file.readlines():
        data = data.strip()
        stopWords.append(data)

stoplst = [' ', '\xa0']
for words in stoplst:
    stopWords.append(words)

Text1=[]
for i in range(len(df['title1_zh'])):
	try:
		poss = jieba.cut(df['title1_zh'][i], cut_all = False)
		Text1.append([])
		for w in poss:
			if w not in stopWords:
				Text1[-1].append(w) 
                #print("W:",w)       
            # if names.get(w) is None and w not in stopWords:    
            #     relationships[w] = {}            
	except:
		pass

Text2=[]
for i in range(len(df['title2_zh'])):
	try:
		poss2 = jieba.cut(df['title2_zh'][i], cut_all = False)
		Text2.append([])
		for w in poss2:
			if w not in stopWords:
				Text2[-1].append(w) 
                #print("W:",w)       
            # if names.get(w) is None and w not in stopWords:    
            #     relationships[w] = {}            
	except:
		pass
print("Text1",Text1[0])
print("Text2",Text2[0])

	



import nltk
import numpy as np
import pandas as pd
import json
import re
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec 

model = Word2Vec.load("word2vec.model")
wordvectors = model.wv

vectors = np.zeros((len(wordvectors.vocab),300))
words = {}
word_count=0
for i in range(len(wordvectors.vocab)):
    words[wordvectors.index2entity[i]]=i
    vectors[i]=wordvectors[wordvectors.index2entity[i]]


# Util function to convert list of strings to one string   
def listToString(s):   
    # initialize an empty string
    str1=""
    
    if(len(s) > 0):
        str1 = s[0] 
    
    # traverse in the string   
    for ele in s[1:]:  
        str1 += " " + ele   
    
    # return string   
    return str1     


#loads query data, needs to be done 
def getCentroidQueryVectors(file):
    file=open(file)
    queryTokens=[]
    count=0
    while True: 
        # Get next line from file 
        line = file.readline() 
        # if line is empty 
        # end of file is reached 
        if not line: 
            break
        count += 1
        queryTokens.append(set(re.split('; |, |"|&|-| |_|\*|\n|\]|\[',line)[1:]))
    
    queryVectors = np.zeros((count,300))
    for i in range(len(queryTokens)):
        cnt=0
        for t in queryTokens[i]:
            if(t.lower() in words):
                cnt+=1
                queryVectors[i] = queryVectors[i] + wordvectors[t.lower()]
        queryVectors[i] = queryVectors[i]/cnt
    return queryVectors


def getCentroidTableVectors(file):
    file=open("data/tables/re_tables-0001.json")
    data = json.load(file)
    file.close()

    tableTokens=[]
    cnt = 0
    for t in data:
        cnt+=1
        #print(t)
        #for k,v in data[t].items():
        #   print ()
        tokens = listToString(data[t]["title"]);
        for d in data[t]["data"]:
            tokens = tokens + " " + listToString(d)
        tokens = tokens + " " + data[t]["caption"]
        tableTokens.append(set(re.split('; |, |"|&|-| |\||_|\*|\n|\]|\[',tokens)))
        tokens = ""

    tableVectors = np.zeros((len(tableTokens),300))
    for i in range(len(tableTokens)):
        cnt=0
        for t in tableTokens[i]:
            if(t.lower() in words):
                cnt+=1
                tableVectors[i] = tableVectors[i] + wordvectors[t.lower()]
        tableVectors[i] = tableVectors[i] / cnt
    #print(tableVectors[2])
    return tableVectors

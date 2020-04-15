import nltk
import numpy as np
import pandas as pd
import json
import re
import math 
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

#computes TFIDF
def TFIDF(td, tc, N):
    return td * math.log(N/tc)

#loads query data, needs to be done 
def getCentroidQueryVectors(file):
    file=open(file)
    queryTokens=[]
    queryTokenCount=[]
    DocTokens={}
    count=0
    while True: 
        # Get next line from file 
        line = file.readline() 
        # if line is empty 
        # end of file is reached 
        if not line: 
            break
        queryTokenCount.append({})
        
        queryTokens.append(set(re.split('; |, |"|&|-| |_|\*|\n|\]|\[',line)[1:]))
        for t in re.split('; |, |"|&|-| |_|\*|\n|\]|\[',line)[1:]:
            if t in queryTokenCount[count]:
                queryTokenCount[count][t]+=1
            elif t in DocTokens:
                DocTokens[t]+=1
                queryTokenCount[count][t]=1
            else:
                DocTokens[t]=1
                queryTokenCount[count][t]=1
        
        count += 1
                
    queryVectors = np.zeros((count,300))
    cnt2=0
    for i in range(len(queryTokens)):
        cnt=0
        for t in queryTokens[i]:
            if(t.lower() in words):
                cnt+=1
                queryVectors[i] = queryVectors[i] + wordvectors[t.lower()] * TFIDF(queryTokenCount[cnt2][t],DocTokens[t],count) 
        cnt2+=1
        queryVectors[i] = queryVectors[i]/cnt
    return queryVectors


def getCentroidTableVectors(file):
    file=open("data/tables/re_tables-0001.json")
    data = json.load(file)
    file.close()
    tableTokens=[]
    tokensHeader = {}
    tokensData = {}
    tokensCaption = {}
    tableTokenCount=[]
    DocTokens={}

    count = 0
    for t in data:
        #print(t)
        #for k,v in data[t].items():
        #   print ()
        tokens = listToString(data[t]["title"]);
        for tok in data[t]["title"]:
                if tok in tokensHeader:
                    tokensHeader[tok]=tokensHeader[tok]+1
                else:
                    tokensHeader[tok]=1
        for d in data[t]["data"]:
            tokens = tokens + " " + listToString(d)
            for tok in d:

                if tok in tokensData:
                    tokensData[tok]=tokensData[tok]+1
                else:
                    tokensData[tok]=1
        tokens = tokens + " " + data[t]["caption"]
        tableTokenCount.append({})
        for tok in re.split('; |, |"|&|-| |\||_|\*|\n|\]|\[',tokens):
            if tok in tableTokenCount[count]:
                tableTokenCount[count][tok]+=1
            elif tok in DocTokens:
                DocTokens[tok]+=1
                tableTokenCount[count][tok]=1
            else:
                DocTokens[tok]=1
                tableTokenCount[count][tok]=1

        tableTokens.append(set(re.split('; |, |"|&|-| |\||_|\*|\n|\]|\[',tokens)))
        tokens = ""
        count+=1
    tableVectors = np.zeros((len(tableTokens),300))
    cnt2=0
    for i in range(len(tableTokens)):
        cnt=0
        for t in tableTokens[i]:
            if(t.lower() in words):
                cnt+=1
                tableVectors[i] = tableVectors[i] + wordvectors[t.lower()] * TFIDF(tableTokenCount[cnt2][t],DocTokens[t],count)
        cnt2+=1
        tableVectors[i] = tableVectors[i] / cnt
    #print(tableVectors[2])
    return tableVectors

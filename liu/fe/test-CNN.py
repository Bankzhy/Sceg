# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 11:32:28 2018

@author: xzf0724
"""
import json
from pathlib import Path

import numpy as np
import time
import os

from sklearn.metrics import classification_report

np.random.seed(1337)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json  
from sklearn import metrics

MAX_SEQUENCE_LENGTH = 15 


# projects=['android-backup-extractor-20140630',"AoI30","areca-7.4.7","freeplane-1.3.12","grinder-3.6","jedit","jexcelapi_2_6_12","junit-4.10","pmd-5.2.0","weka"]
def camel_case_split(str):
    if len(str) == 0:
        return ""
    words = [[str[0]]]

    for c in str[1:]:
        if words[-1][-1].islower() and c.isupper():
            words.append(list(c))
        else:
            words[-1].append(c)

    return [''.join(word) for word in words]

def under_case_split(w):
    ws = w.split("_")
    return ws

def split_token(word):

    if "_" in word:
        result = under_case_split(word)
    else:
        result = camel_case_split(word)
    return result

def load_data(path, label):
    distances = []  # TrainSet
    labels = []  # 0/1
    texts = []  # ClassNameAndMethodName

    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            fe_graph = json.loads(line)
            nodes = fe_graph["nodes"]
            dist = []
            text = []
            for index, node in enumerate(nodes):
                if node["type"] == "class":
                    dist.append(node["metrics"]["dist"])
                    class_name_text = split_token(node['name'])
                    class_name_text = class_name_text.join(" ")
                    class_name_text = class_name_text.lower()
                    text.append(class_name_text)
                elif node["type"] == "method":
                    method_name_text = split_token(node['name'])
                    method_name_text = method_name_text.join(" ")
                    method_name_text = method_name_text.lower()
                    text.append(method_name_text)
            distances.append(dist)
            texts.append(text)
            labels.append(label)
    return distances, texts, labels

def test():
    ttp,ttn,tfp,tfn=0,0,0,0

    auc_y=np.zeros(shape=(0))
    auc_p=np.zeros(shape=(0))


    # for ith in range(len(projects)):#len(projects)
        
    # project=projects[ith]
    ss=time.time()

    # print('-'*80)
    # print(project)


    # TESTPATH = 'D:/TSE/python/featureenvy/data/'+project+'/'
    MODELPATH = ''
    # FILENAME = 'D:/TSE/python/featureenvy/data/'+project+'/test_ClssId.txt'
    # TARGETPATH = 'D:/TSE/python/featureenvy/data/'+project+'/targetClasses.txt'


    # values = []
    # predsTargetClassNames = []
    # print ("start time:"+time.strftime("%Y/%m/%d  %H:%M:%S"))
    # start = time.clock()
    # f = open(TARGETPATH, 'r', encoding = 'utf-8')
    # for line in f:
    #     predsTargetClassName = line.split()
    #     predsTargetClassNames.append(predsTargetClassName)
    #
    # f = open(FILENAME, 'r', encoding = 'utf-8')
    # for line in f:
    #     value = line.split()
    #     values.append(value)
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    NUM_CORRECT = 0
    TOTAL = 0

    models=[]

    # for index in range(5):################################################################
    #     #index=3
    model=model_from_json(open('fe.json').read())
    model.load_weights('fe.h5')
    # models.append(t)
    ii = 0

    tauc_y=[]
    tauc_p=[]

    test_distances = []
    test_labels = []
    test_texts = []
    targetClassNames=[]

    pos_path = Path("dataset/fe/" + "test" + "_1.txt")
    pos_dist, pos_text, pos_label = load_data(pos_path, 1)
    test_distances.extend(pos_dist)
    test_texts.extend(pos_text)
    test_labels.extend(pos_label)

    neg_path = Path("dataset/fe/" + "test" + "_0.txt")
    neg_dist, neg_text, neg_label = load_data(neg_path, 0)
    test_distances.extend(neg_dist)
    test_texts.extend(neg_text)
    test_labels.extend(neg_label)

    # with open(TESTPATH + 'test_Distances'+classId+'.txt','r') as file_to_read:
    #     for line in file_to_read.readlines():
    #         values = line.split()
    #         test_distance = values[:2]
    #         test_distances.append(test_distance)
    #         test_label =values[2:]
    #         test_labels.append(test_label)
    #
    #
    # with open(TESTPATH + 'test_Names'+classId+'.txt','r') as file_to_read:
    #     for line in file_to_read.readlines():
    #         test_texts.append(line)
    #         line = line.split()
    #         targetClassNames.append(line[10:])

    tokenizer1 = Tokenizer(num_words=None)
    tokenizer1.fit_on_texts(test_texts)
    test_sequences = tokenizer1.texts_to_sequences(test_texts)
    test_word_index = tokenizer1.word_index
    test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    test_distances = np.asarray(test_distances)
    test_labels1 = test_labels
    test_labels = np.asarray(test_labels)


    x_val = []
    x_val_names = test_data
    x_val_dis = test_distances
    x_val_dis = np.expand_dims(x_val_dis, axis=2)
    x_val.append(x_val_names)
    x_val.append(np.array(x_val_dis))
    y_val = np.array(test_labels)

    preds = model.predict_classes(x_val)
    target_names = ["neg", "pos"]
    print(classification_report(y_val, preds, target_names=target_names))


    # preds_doubles=[]#------------------
    #
    #
    # total_pp=[]
    # result=[]
    # for index in range(5):######################################################
    #     pp=models[index].predict(x_val)
    #     total_pp.append(pp)
    #     preds_doubles.append(pp)#--------------------------
    #     tpp=np.argmax(pp,axis=1)
    #     result.append(tpp)
    #
    # auc_temp=-100
    #
    # for i in range(y_val.shape[0]):
    #     p0,p1=0.0,0.0
    #     for j in range(5):################################################################
    #         p0+=total_pp[j][i][0]
    #         p1+=total_pp[j][i][1]
    #     if (p1-p0)/5>auc_temp:
    #         auc_temp=(p1-p0)/5
    #
    # tauc_p.append(auc_temp)
    # if label=='0':
    #     tauc_y.append(0)
    # else:
    #     tauc_y.append(1)
    #
    # tpreds=[]
    #
    # for i in range(len(result[0])):
    #     total=0
    #     for j in range(5):################################################
    #         total=total+result[j][i]
    #     if total>=3:#####################################################
    #         tpreds.append(1)
    #     else:
    #         tpreds.append(0)
    # preds=np.array(tpreds)
    #
    # #-------------------------------------
    # NUM_ZERO = 0
    # NUM_ONE = 0
    # for i in range(len(preds)):
    #     if(preds[i]==0):
    #         NUM_ZERO += 1
    #     else:
    #         NUM_ONE += 1
    # if(len(preds)!=0 and label == '1'):
    #     TOTAL+=1
    # if(label == '1' and NUM_ONE == 0):
    #     FN += 1
    # if(label == '1' and NUM_ONE != 0):
    #     TP+=1
    #     preds_double=[]
    #     for kk in range(len(preds_doubles[0])):
    #         ttt=0.0
    #         for kt in range(5):#################################################
    #             ttt+=preds_doubles[kt][kk][1]
    #         preds_double.append(ttt)
    #
    #     correctTargets = []
    #
    #
    #     MAX=0
    #     for i in range(len(preds_double)):
    #         if(preds_double[i]>=MAX):
    #             MAX = preds_double[i]
    #     for i in range(len(preds_double)):
    #         if(preds_double[i] == MAX):
    #             correctTargets.append(targetClassNames[i])
    #             break
    #
    #     for i in range(len(correctTargets)):
    #         if(correctTargets[i]==predsTargetClassNames[TOTAL-1]):
    #             NUM_CORRECT += 1
    #             break
    # if(label == '0' and NUM_ONE == 0):
    #     TN += 1
    # if(label == '0' and NUM_ONE !=0):
    #     FP += 1

    # print('TP--------', TP)
    # print('TN--------', TN)
    # print('FP--------', FP)
    # print('FN--------', FN)
    # print('NUM_ZERO---', NUM_ZERO)
    # print('NUM_ONE---', NUM_ONE)
    # print('NUM_CORRECT----',NUM_CORRECT)
    # print('TargetAccuracy---',NUM_CORRECT/TP)
    # tauc_y=np.array(tauc_y)
    # tauc_p=np.array(tauc_p)
    # auc_y=np.concatenate((auc_y,tauc_y),axis=0)
    # auc_p=np.concatenate((auc_p,tauc_p),axis=0)
    # print("AUC : ",metrics.roc_auc_score(tauc_y.astype(int),tauc_p))
    #
    # precision=TP*1.0/(TP+FP)
    # recall=TP*1.0/(TP+FN)
    # f1=2*precision*recall/(precision+recall)
    # print(precision,recall,f1)
    #
    # ttp+=TP
    # ttn+=TN
    # tfp+=FP
    # tfn+=FN
    # print('########################',time.time()-ss)
    #
    #
    # print("--------------------------------Final")
    # print(ttp,ttn,tfp,tfn)
    # precision=ttp*1.0/(ttp+tfp)
    # recall=ttp*1.0/(ttp+tfn)
    # f1=2*precision*recall/(precision+recall)
    # print(precision,recall,f1)
    # print("AUC : ",metrics.roc_auc_score(auc_y.astype(int),auc_p))
    # print("--------------------------------")

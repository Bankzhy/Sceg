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
                    class_name_text = " ".join(class_name_text)
                    class_name_text = class_name_text.lower()
                    text.append(class_name_text)
                elif node["type"] == "method":
                    method_name_text = split_token(node['name'])
                    method_name_text = " ".join(method_name_text)
                    method_name_text = method_name_text.lower()
                    text.append(method_name_text)
            distances.append(dist)
            texts.append(text)
            labels.append(label)
    return distances, texts, labels

def test():
    MODEL_NUMBER = 5

    test_distances = []
    test_labels = []
    test_texts = []


    pos_path = Path("../../dataset/fe/liu/" + "test" + "_1.txt")
    pos_dist, pos_text, pos_label = load_data(pos_path, 1)
    test_distances.extend(pos_dist)
    test_texts.extend(pos_text)
    test_labels.extend(pos_label)

    neg_path = Path("../../dataset/fe/liu/" + "test" + "_0.txt")
    neg_dist, neg_text, neg_label = load_data(neg_path, 0)
    test_distances.extend(neg_dist)
    test_texts.extend(neg_text)
    test_labels.extend(neg_label)

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

    preds = []
    # preds = model.predict_classes(x_val)
    for index in range(MODEL_NUMBER):
        model = model_from_json(open("fe"+str(index)+".json").read())
        model.load_weights('fe'+str(index)+'.h5')
        predict_prob = model.predict(x_val)
        pred = np.argmax(predict_prob, axis=1)
        preds.append(pred)
    result = []
    for i in range(len(preds[0])):
        total = 0
        for j in range(MODEL_NUMBER):
            total = total + preds[j][i]
        if total >= 3:
            result.append(1)
        else:
            result.append(0)


    target_names = ["neg", "pos"]
    print(classification_report(y_val, result, target_names=target_names))

test()
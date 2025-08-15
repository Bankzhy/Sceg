import json
import random

import joblib
from sklearn import preprocessing,metrics
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn import neighbors
from sklearn import svm

import time

MODEL_NUMBER=5
SUBSET_SIZE=0.8

tr=[]
trp=[]


def eval(tp, tn, fp, fn):
    print("tp : ", tp)
    print("tn : ", tn)
    print("fp : ", fp)
    print("fn : ", fn)
    P = tp * 1.0 / (tp + fp)
    R = tp * 1.0 / (tp + fn)
    print("Precision : ", P)
    print("Recall : ", R)
    print("F1 : ", 2 * P * R / (P + R))
    if tp == 0 or tn == 0 or fp == 0 or fn == 0:
        return 1

    a = tp + fp
    b = tp + fn
    c = tn + fp
    d = tn + fn
    print("MCC : ", (tp * tn - fp * fn) / ((a * b * c * d) ** 0.5))

    return 2 * P * R / (P + R)


def load_dataset(split):
    pos_path = r"../../dataset/lm/"+split+"_1.txt"
    neg_path = r"../../dataset/lm/"+split+"_0.txt"

    features = []
    labels = []

    with open(pos_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            new_graph = json.loads(line)
            nodes = new_graph["nodes"]
            feature = []
            for node in nodes:
                if "type" in node.keys():
                    feature.append(node["metrics"]["loc"])
                    feature.append(node["metrics"]["lcom1"])
                    feature.append(node["metrics"]["lcom2"])
                    feature.append(node["metrics"]["lcom4"])
                    feature.append(node["metrics"]["coh"])
                    feature.append(node["metrics"]["clc"])
                    feature.append(node["metrics"]["noav"])
                    feature.append(node["metrics"]["cd"])
                    feature.append(node["metrics"]["cc"])
                    break
            features.append(feature)
            labels.append(1)

    with open(neg_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            new_graph = json.loads(line)
            nodes = new_graph["nodes"]
            feature = []
            for node in nodes:
                if "type" in node.keys():
                    feature.append(node["metrics"]["loc"])
                    feature.append(node["metrics"]["lcom1"])
                    feature.append(node["metrics"]["lcom2"])
                    feature.append(node["metrics"]["lcom4"])
                    feature.append(node["metrics"]["coh"])
                    feature.append(node["metrics"]["clc"])
                    feature.append(node["metrics"]["noav"])
                    feature.append(node["metrics"]["cd"])
                    feature.append(node["metrics"]["cc"])
                    break
            features.append(feature)
            labels.append(0)
    # Zip, shuffle, and unzip
    combined = list(zip(features, labels))
    random.shuffle(combined)
    features, labels = zip(*combined)

    # Convert back to lists (optional)
    features = list(features)
    labels = list(labels)
    return features, labels

def train():
    # data_path = "/Volumes/rog/research/example/DeepSmellDetection-master/long method/Data Generation/dm_result.csv"
    # df = pd.read_csv(data_path)
    # data0=df[(df.label==0)]
    # data1=df[(df.label==1)]
    # num = (int)(data1.shape[0] * SUBSET_SIZE)

    models = []
    for i in range(MODEL_NUMBER):
        print("training ", i + 1, "th model")
        # data0 = shuffle(data0)
        # data1 = shuffle(data1)
        #
        # train_set0 = data0.iloc[:num, :]
        # train_set1 = data1.iloc[:num, :]
        #
        # data = train_set0.append(train_set1)
        # data = shuffle(data)
        #
        # x = data.iloc[:, 4:13]
        # y = np.array(data.iloc[:, 3])
        x, y = load_dataset("train")

        # x=preprocessing.scale(x,axis=1,with_mean=True,with_std=True,copy=True)

        clf = MLPClassifier(hidden_layer_sizes=(16, 8, 4), max_iter=200, )

        # clf=RandomForestClassifier(n_estimators=100)

        clf.fit(x, y)

        # save model
        joblib.dump(clf, "liu-v1" + "_" + str(i) + ".joblib")

        models.append(clf)

    return models


def test_one(model):
    # df = pd.read_csv(data_path,
    #                  encoding='ISO-8859-1')
    clf = model
    # x = df.iloc[:, 4:13]
    x, y = load_dataset("test")
    predict = clf.predict(x)
    predict_proba = clf.predict_proba(x)

    # y = np.array(df.iloc[:, 3])

    tp, tn, fp, fn = 0, 0, 0, 0

    for i in range(len(y)):
        tr.append(y[i])
        if predict[i] == y[i]:
            if predict[i] == 0:
                tn = tn + 1
            else:
                tp = tp + 1
        else:
            if predict[i] == 0:
                fn = fn + 1
            else:
                fp = fp + 1

    return tp, tn, fp, fn


def test(models):
    # data_path = "/Volumes/rog/research/example/DeepSmellDetection-master/long method/Data Generation/dm_60_result.csv"
    # df = pd.read_csv(data_path,
    #                  encoding='ISO-8859-1')

    # df = df[(df.projectname == projectName)]
    predicts = []
    predicts_proba = []
    x, y = load_dataset("test")
    for i in range(MODEL_NUMBER):
        clf = models[i]
        # x = df.iloc[:, 4:13]

        # x=preprocessing.scale(x,axis=1,with_mean=True,with_std=True,copy=True)

        predict = clf.predict(x)
        predict_proba = clf.predict_proba(x)

        predicts.append(predict)
        predicts_proba.append(predict_proba)
    result = []
    for i in range(len(predicts[0])):
        total = 0
        for j in range(MODEL_NUMBER):
            total = total + predicts[j][i]
        if total >= 3:
            result.append(1)
        else:
            result.append(0)

    rp = []
    for i in range(len(predicts_proba[0])):
        total = 0
        for j in range(MODEL_NUMBER):
            total = total + predicts_proba[j][i][1]
        rp.append(total / MODEL_NUMBER)
        trp.append(total / MODEL_NUMBER)

    # y = np.array(df.iloc[:, 3])
    target_names = ["neg", "pos"]
    print(classification_report(y, result, target_names=target_names))
    print('*' * 80)
    print("AUC : ", metrics.roc_auc_score(y, rp))
    print('*' * 80)
    tp, tn, fp, fn = 0, 0, 0, 0

    for i in range(len(y)):
        tr.append(y[i])
        if result[i] == y[i]:
            if result[i] == 0:
                tn = tn + 1
            else:
                tp = tp + 1
        else:
            if result[i] == 0:
                fn = fn + 1
            else:
                fp = fp + 1

    return tp, tn, fp, fn


def load_models():
    models=[]

    for i in range(MODEL_NUMBER):
        #clf=joblib.load('D:/Longmethod/model/4677/'+projectName+"_"+str(i)+'.joblib')
        clf=joblib.load("liu-v1" + "_" + str(i) + ".joblib")

        models.append(clf)

    return models


ttp,ttn,tfp,tfn=0,0,0,0
for i in range(1):

    print("------------------------------------")
    ss=time.time()
    model=train()
    print('#####################', time.time()-ss)
    models = load_models()
    ss=time.time()
    tp,tn,fp,fn=test(models)
    # tp, tn, fp, fn = test_one(models[0])

    print(time.time()-ss)
    ttp=ttp+tp
    ttn=ttn+tn
    tfp=tfp+fp
    tfn=tfn+fn
    eval(tp,tn,fp,fn)
print("------------------------------------")
print("Final Evaluation:")
ans=eval(ttp,ttn,tfp,tfn)
print("AUC : ",metrics.roc_auc_score(tr,trp))

import pylab as pl
import numpy as np
import numpy.linalg as la
import math
import csv

def loadCsv(f):
    lines = csv.reader(open(f,"rb")) 
    data = list(lines)
    #for i in range(len(data)):
    #    data[i] = [float(di) for di in data[i]]
    return data


def split(data, splitratio):
    train_size = len(data)*splitratio
    test = list(data)
    train = [] 
    while len(train) < train_size:
        index = np.random.randint(0,len(test))
        train.append(test.pop(index))
    return train, test


def separateByClass(data):
    classes = {}
    for i in range(len(data)):
        v = data[i]
        if v[-1] not in classes.keys():
           classes[v[-1]] = []
        classes[v[-1]].append(v[:-1])
    return classes

def buildModel(data):
    classes = separateByClass(data)
    wordProbs = []
    prior = []
    for c in classes.keys():
        counts = {} 
        class_data = classes[c]
        for i in class_data:
            for j in i:
               if j not in counts.keys():
                   counts[j] = 0
               counts[j] += 1
        py = len(class_data)
        for i in counts.keys():
            counts[i] = float(counts[i]+1) / (py + 2)
        wordProbs.append(counts)
        prior.append(py)
    return wordProbs, prior


def predict(test, probs, prior):
    predictions = []
    test_features = []
    testVals = []
    for t in test:
        test_features.append(t[:-1])
        testVals.append(t[-1])

    for t in test_features:
        pxy = []
        for p,q in zip(probs, prior):
            pred = 1
            for term in t:
                if term in p.keys():
                   pred *= p[term]
                else:
                   pred *= 0.5
            pred = pred*q
            pxy.append(pred)
        max_val = max(pxy)
        arg_max = pxy.index(max_val)
        predictions.append(arg_max)

    testVals = np.asarray ([int(x) for x in testVals] )
    predictions = np.asarray ([int(x) for x in predictions] )

    acc = (testVals == predictions)
    print testVals, predictions, acc
    acc = np.asarray([ 1 if x == True else 0 for x in acc])
    print testVals, predictions, acc

    #acc = np.asarray(testVals * predictions)
    
    print "Accuracy : " , float(acc.sum())/len(acc)
    
    
 

if __name__ == '__main__':

    filename = 'tmp.csv'
    splitratio = 0.7
    data = loadCsv(filename)
    train, test = split(data, splitratio)
    probs, priors = buildModel(train)

    #print  probs

    testPred = predict(test, probs, priors) 
    
     


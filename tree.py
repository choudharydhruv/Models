import math
import numpy as np
import pylab as pl
import numpy.linalg as la

class NodeArray(object):

    def __init__(self,size=0):
    self.size = 0
    

class DecisionTree(object):

    def __init__(self,metric='entropy', depth=4):
        self.metric = metric
        self.depth = depth

    def getMetric(self,p):
        if self.metric == entropy:
            return -1*p*np.log(p)


    def getNodeImpurity(self, Y, num_classes):
        total_impurity = 0
        for i in range(num_classes):
            p = (Y==i)
            p = np.asarray([1 if x is True else 0 for x in p])
            if len(p) >0
              p = float(p.sum())/len(p)
              total_impurity += self.getMetric(p)
        return total_impurity

    def findBestSplitInVariable(x,y):
        classes_x = {}
        classes_y = {} 
        for i in range(len(x)):
            if x[i] not in classes_x.keys():
                classes_x[x[i]] = []
                classes_y[x[i]] = []
            classes_x[x[i]].append(x[i])
            classes_y[x[i]].append(y[i])
        impurity = [] 
        for c in classes_y.keys():
            impurity.append( getNodeImpurity( classes_y[c],2) )

        min_impurity = max(impurity)
        index = impurity.index(min_impurity)
         
        return min_impurity, index, classes_x[index], classes_y[index]


    def findBestSplit(X, y, d):
        num_var = X.shape[1]
        impurities = []
        var_class_vals = []
        xdata_list = []
        ydata_list = []
        for i in range(num_var):
            x = X[:,i]
            x_impurity, x_val, xdata, ydata = findBestSplitInVariable(x,y)
            impurities.append(x_impurity)
            var_class_vals.append(x_val)
            xdata_list.append(xdata)
            ydata_list.append(ydata)
         
        min_impurity = max(impurities)
        index = impurities.index(min_impurity)



    def fit(X,y):
        X1,X2,y1,y2 = findBestSplit(X,y,self.depth)


  




if __name__ == '__main__':
    


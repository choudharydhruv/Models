import pylab as pl
import numpy as np
import scipy.cluster.vq as vq
from scipy.stats import norm

class AnamolyDetection(object):
    
    def __init__(self, type='gaussian'):
        self.type = type

    def density(self, v, idx):
        d = ( ((v-self.dim_mean[idx]) ** 2) / (2*(self.dim_std[idx] ** 2) ))
        print "Scipy density " , norm.pdf(d) 
        d = np.exp(-1*d) / (np.sqrt(2*np.pi)*self.dim_std[idx])
        #d = [x if x>1e-9 else 1e-5  for x in d]
        print "Individual density ", d
        #print "Scipy density " , norm.pdf(v, loc=self.dim_mean[idx], scale=self.dim_std[idx]) 
        return d

    def udensity(self, V):
        m,n = V.shape
        d = np.ones(m)
        print m,n
        for i in range(n):
            d = d*self.density(V[:,i], i)    
        return d   

    def mdensity(self, X):
        m,n = X.shape
        sigma_inv = np.linalg.inv(self.sigma)
        de = []
        #For calculating mdensity you have to break down matrix into each sample by sample
        for x in X: 
            d = np.exp( -0.5* np.dot(x,sigma_inv).dot(x.T) )
            d = d / ( ( (2*np.pi) ** 0.5) * np.sqrt(np.linalg.det(self.sigma) ) )
            de.append(d)
        return de 

    def cov_matrix(self,X):
        return np.dot(X.T, X)/X.shape[0]

    def fit(self, X):
        m,n = X.shape
        self.dim_mean = np.zeros(n)
        self.dim_std = np.zeros(n)
        for i in range(n):
            self.dim_mean[i] = X[:,i].mean(0)
            self.dim_std[i] = X[:,i].std(0)
        print self.udensity(X)
        self.sigma = self.cov_matrix(X) # can also use np.cov
        print self.mdensity(X)

    def detect(self,X):
        print self.udensity(X)


if __name__ == '__main__':

    m=20 
    n=2

    X = 2.5*np.random.randn(m,n) + 5
    #print X
    ad = AnamolyDetection()
    ad.fit(X)
    
    



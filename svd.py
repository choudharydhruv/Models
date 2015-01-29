import pylab as pl
import numpy as np




def cov(X):
    return np.dot(X.T, X)/X.shape[0]

class PCA(object):

    def __init__(self, k):
        self.k = k

    def fit(self, X):
        cov_matrix = cov(X)

        U,s,V = np.linalg.svd(cov_matrix, full_matrices=True)

        self.Ureduce = U[:,0:k-1]
        print self.Ureduce.shape, X.shape       
 
        Xreduce = np.dot(self.Ureduce.T, X.T);

        var_red = 1- (s[0:k].sum()/ s.sum())
        print s
        print var_red

        cs = np.zeros(s.shape)
        cs[0] = s[0]
        for i, sing in enumerate(s[1:]):
             cs[i] = s[i] + cs[i-1]

        pl.plot(cs)
        pl.show()


if __name__ == '__main__':
    m=1000 
    n=100
    k=5
    X = np.random.uniform(-1,1, size=(m,n))
    X = (X-X.mean(0))/X.std(0)
    
    dr = PCA(k=2)
    dr.fit(X)



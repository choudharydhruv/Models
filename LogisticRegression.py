import numpy as np
from scipy import linalg, optimize
from scipy.optimize import check_grad


class LogisticRegression:

    def __init__(self,bayesian=True, loss='log', method='cg', n_features=10):
        self.bayesian = bayesian
        self.loss = loss
        self.method = method
        self.Theta = np.random.randn(n_features)

    def gethThetaX(self, X, Theta):
        w = np.dot(X,Theta)
        #print "Wshape " , w.shape, X.shape, Theta.shape
        w[w<-20.] = -20.
        w[w>20] = 20.
        #hThetaX = 1./ (1. + np.exp(-1*max( min(w, 20.), -20. )) )
        hThetaX = 1./ (1. + np.exp(-1*w))
        return hThetaX

    def logloss(self, Theta, X, y, reg):
        if np.size(Theta.shape)==1:
            Theta = Theta[:,np.newaxis]
        if self.loss == 'log':
            hThetaX = self.gethThetaX(X,Theta)
            l = y*np.log(hThetaX) + (1-y)*np.log(1-hThetaX)
            #print l.shape 
            loss = l.mean() + reg*np.sum(np.dot(Theta.T,Theta))/X.shape[0]
            return -1*loss #Always remember loss is negative and grad is hthetax-y but positive. 

    def gradient(self, Theta, X, y, reg):
        if np.size(Theta.shape)==1:
            Theta = Theta[:,np.newaxis]
        #print "Grad", X.shape, Theta.shape
        hThetaX = self.gethThetaX(X, Theta)
        g = (hThetaX-y)
        t = Theta
        t[0] = 0.
        #print X.shape, y.shape, hThetaX.shape, g.shape, t.shape
        grad = np.dot(X.T, g) + reg*t 
        grad = grad/X.shape[0]
        return grad

    # Hessian is (1/m)* -1*XT W X where W is diagonal matrix of xi*xiT covariance values
    def hessian(self, Theta, X, y):
        hThetaX = self.gethThetaX(X, Theta)
        h = hThetaX * (1-hThetaX)

        #Creating a diagonal matrix W with the diagonal as hThetaX * (1-hThetaX) 
        W = np.eye(X.shape[0],X.shape[0])
        for i in range(0,X.shape[0]):
            W[i][i] = h[i]
        #print h,W
        h = np.dot(X.T,W.T)
        h = np.dot(h,X)
        return -1*h/X.shape[0]

    def cg(self, X,Theta,y,reg):
        print "original logloss " , self.logloss(Theta, X, y,reg)
        t = optimize.fmin_cg(self.logloss, Theta, fprime=self.gradient, args = (X,y,reg), gtol=1e-6,maxiter=200)
        self.Theta = t
        print "final logloss " , self.logloss(self.Theta, X, y,reg)

    def grad_desc(self, X, Theta, y, reg):
        for i in range(1,200):
            #print "Logloss " , self.logloss(Theta, X, y,reg)
            print "Hessian " , self.hessian(Theta, X, y)
            Theta -= 0.3*self.gradient(Theta,X,y,reg) 

    def newton_raphson(self, X, Theta, y , reg):
        for i in range(1,10):
            print "Logloss " , self.logloss(Theta, X, y,reg)
            #print "Hessian " , self.hessian(Theta, X, y)
            H = self.hessian(Theta, X, y)
            hinv = np.linalg.pinv(H)
            grad = self.gradient(Theta,X,y,reg)
            #print hinv.shape, grad.shape
            Theta -= np.dot(hinv,grad)

    def fit(self, X, Theta, y, reg):
        if self.method=='cg':
            self.cg(X,Theta,y,reg)
        if self.method=='gd':
            self.grad_desc(X,Theta,y,reg)
        if self.method=='nw':
            self.newton_raphson(X,Theta,y,reg)

    def predict(self, Xnew, ynew):
        hThetaX = self.gethThetaX(Xnew, self.Theta)
        print "Misclassification rate ", (ynew*(hThetaX>0.5)).mean()
        return hThetaX

if __name__ == '__main__':
    for corr in (0., 1., 10.):
        m,n = 10**2, 10**1
        np.random.seed(0)
        X = np.random.randn(m,n)
        X += 0.8*np.random.randn(n)
        X += corr
        X = np.hstack( (np.ones( (X.shape[0], 1)), X) )
        Theta = np.random.randn(n+1,1)
        y = np.sign(np.dot(X,Theta))
        y= (y+1)/2
        #print y[0:5]
        print X.shape, Theta.shape, y.shape
        reg = 0.1

        lr = LogisticRegression(loss='log', method='nw')
        lr.fit(X, Theta, y, reg)
    




    #check_grad(logloss,gradient,[])












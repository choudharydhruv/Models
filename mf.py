import pylab as pl
import numpy as np
import scipy.cluster.vq as vq

def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in xrange(steps):
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = np.dot(P,Q)
        e = 0
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in xrange(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        if e < 0.001:
            print "Used iterations ", step
            break
    err = np.dot(P,Q)
    print err
    return P, Q.T


def matrixFactorization2(R, Theta, X, K, steps=5000, alpha=0.0002, lamb=0.02):
    X = X.T # Now the two matrices are (Nu x k) and (k x Nm)
    err = np.dot(Theta, X)
    Nu, Nm = R.shape
    for step in xrange(steps):
        for i in xrange(Nu):
            for j in xrange(Nm):
                if R[i][j] >0:
                    eij = np.dot(Theta[i,:],X[:,j]) -R[i][j]
                    for k in xrange(K):
                        Theta[i][k] -= alpha * (eij * X[k][j] + lamb * Theta[i][k])
                        X[k][j] -= alpha * (eij * Theta[i][k] +lamb * X[k][j])  
    
        #print R.shape, Theta.shape, err.shape, X.shape
        #Theta = Theta - alpha*( np.dot(err,X) + lamb * Theta)
        #X = X - alpha*( np.dot(err.T, Theta) + lamb * X)
        err = np.dot(Theta, X)
    print err
    return Theta, X.T




if __name__ == "__main__":
    R = [
         [5,3,0,1],
         [4,0,0,1],
         [1,1,0,5],
         [1,0,0,4],
         [0,1,5,4],
        ]

    R = np.array(R)

    N = len(R)
    M = len(R[0])
    K = 2

    Theta = np.random.rand(N,K)
    X = np.random.rand(M,K)

    nTheta, nX = matrixFactorization2(R, Theta, X, K)
    nTheta, nX = matrix_factorization(R, Theta, X, K)

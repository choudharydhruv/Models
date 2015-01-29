import numpy as np
from scipy.stats import norm
import pylab as pl



max_iter = 100


m=100
a=0.3

s1 = 0.08 * np.random.randn(m*a)
s2 = 0.6 + 0.12*np.random.randn(m*(1-a))
print s1.shape, s2.shape

s = np.concatenate([s1,s2])

def norm_pdf(x,u,s):
    return np.exp(-1*( (x-u)**2 ) / (2*(s**2)) ) / ( np.sqrt(2*np.pi)*s)


def pdf_model(x,p):
    u1, s1,u2,s2, pi = p
    return pi*norm_pdf(x,u1,s1) + (1-pi)*norm_pdf(x,u2,s2)


# 
p = np.asarray([-0.2,0.2, 0.8,0.2,0.5])

u1, v1, u2, v2, pi = p
U = np.asarray([u1, u2])
V = np.asarray([v1, v2])
PI = np.asarray([pi, 1-pi])

gamma = np.zeros((m,2))
N = np.zeros((2,1))

max_iter = 5

converged = False
while max_iter>0:
    for k in range(2):
        #print gamma[:,k].shape, s.shape, U[k].shape, V[k].shape
        #print pdf_model(s,p).shape
        #print norm.pdf(s,U[k],V[k]).shape
        #print PI[k].shape

        gamma[:,k] = PI[k]*norm_pdf(s,U[k],V[k]) / pdf_model(s,p) 
        N[k] = gamma[:,k].sum()
        U[k] = (gamma[:,k]*s).sum() / N[k]
        V[k] = np.sqrt( (gamma[:,k]* ((s-U[k])**2) ).sum()/ N[k] )
        PI[k] = N[k]/m
    p = [U[0], V[0], U[1], V[1], PI[0]]
 
    print gamma
    print N.T,U,V,PI
    max_iter -= 1
    





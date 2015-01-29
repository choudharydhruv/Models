import pylab as pl
import numpy as np
import scipy.cluster.vq as vq



np.random.seed(3763736)

def euclid_dist(x1,x2):
    return np.linalg.norm(x1-x2)

def recalculate_clusters(X,u,clusters):
    unew = []
    for i,c in enumerate(u):
        newc = (X[clusters == i]).mean()
	print clusters
        unew.append(newc)
    return unew

def cluster_points(X,u):
    clusters = []
    for x in X:
        closest = 0
        dist = euclid_dist(x,u[0]) 
        for i,c in enumerate(u[1:]):
            dnew = euclid_dist(x,c)
            if dnew < dist:
	        dnew = dist
                closest = i+1
        clusters.append(closest)
    return clusters 

def converged(oldu, u):
    for u1,u2 in zip(oldu,u):
        if euclid_dist(u1,u2)>1e-2:
            return False
    return True

def find_centers(X,k):
    np.random.shuffle(X)
    oldu = X[-k+1:]
    np.random.shuffle(X)
    u = X[0:k-1]
    print u, oldu

    while not converged(u,oldu):
        oldu = u
        clusters = cluster_points(X,u)
        u = recalculate_clusters(X,u,clusters) 
        print u
    return u

m=1000 
n=2
k=5

X = np.random.uniform(-1,1, size=(m,n))

X = (X-X.mean(axis=0))/X.std(axis=0)

#X = np.asarray([2,4,3,19,17,21,91,87,98])
#k=3

#print find_centers(X,k)

cent, label = vq.kmeans2(X,k)
print cent[:,1]



#print X[0], X[1], euclid_dist(X[0],X[1])


cent = np.random.randn(k,n)
cent_new = cent

loc = np.zeros(m)
dist = -1*np.ones((k,m))


iter = 10
biggest_shift = 0 

while(iter>0):
    #print X
    #print cent
    for i in range(0,m-1):
        smallest = euclid_dist(cent[0], X[i])
        loc[i] = 0;
        for j in range(1,k-1):
            d = euclid_dist(cent[j], X[i])
            if d < smallest:
                loc[i] = j

    for i in range(0,k-1):
        cent_new[i,:] = X[loc==i,:].sum()/X.shape[0]
        shift = euclid_dist(cent[i], cent_new[i])
        biggest_shift = max(biggest_shift, shift)

    iter -= 1
    cent = cent_new

    cost =0
    for i in range(0,m-1):
        cost += euclid_dist(X[i], cent[loc[i]]) ** 2

    print cost;

    pl.plot(cent[:,0], cent[:,1], 'ro', X[:,0], X[:,1], 'go')
    pl.show()
    if biggest_shift < 1e-2:
        break 


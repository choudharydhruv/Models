import numpy as np
import cvxopt
import cvxopt.solvers


MIN_SUPPORT_VECTOR_MULTIPLIER = 1e-5

class SVMModel(object):
    def __init__(self,kernel,c):
        self.kernel = kernel
        self.c = c
    
    def fit(self, X, y):
        lagrange = self.get_lagrange_multipliers(X,y)

        support_vector_indices = lagrange> MIN_SUPPORT_VECTOR_MULTIPLIER

        weights = lagrange[support_vector_indices]
        support_vectors = X[support_vector_indices]
        support_vector_labels = y[support_vector_indices]

        bias = np.mean( [y_k - SVMPredict(self.kernel, bias=0.0, weights = weights, support_vectors = support_vectors, support_vector_labels = support_vector_labels).predict(x_k) for y_k, x_k in zip(support_vector_labels, support_vectors) ] )

        return SVMPredict(kernel = self.kernel , bias=bias, weights = weights, support_vectors = support_vectors, support_vector_labels = support_vector_labels)

 
    def gram_matrix(self,X):
        m,n = X.shape
        K = np.zeros((m,m))
        for i, x_i in enumerate(X):
            for(j, x_j) in enumerate(X):
                K[i,j] = self.kernel(x_i,x_j)
        return K

    def get_lagrange_multipliers(self, X, y):
        m,n = X.shape
        K = self.gram_matrix(X)

        # solves 1/2 xT P x + qT x s.t. Gx<h Ax=b 

        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(-1* np.ones(m))

        G_std = cvxopt.matrix(np.diag(np.ones(m) * -1))
        h_std = cvxopt.matrix(np.zeros(m))

        G_slack = cvxopt.matrix(np.diag(np.ones(m)))
        h_slack = cvxopt.matrix(np.ones(m) * self.c)

        G = cvxopt.matrix(np.vstack((G_std, G_slack)))
        h = cvxopt.matrix(np.vstack((h_std, h_slack))) 


        A = cvxopt.matrix(y, (1,m))
        b = cvxopt.matrix(0.0)

        solution = cvxopt.solvers.qp(P,q,G,h,A,b)

        return np.ravel(solution['x'])

class SVMPredict(object):
    def __init__(self,
                  kernel,
                  bias,
                  weights,
                  support_vectors, 
                  support_vector_labels):
        self.kernel = kernel
        self.bias = bias
        self.weights = weights
        self.support_vectors = support_vectors
        self.support_vector_labels = support_vector_labels

    def predict(self,X):
        result = self.bias
        #Adding values of all support vectors
        for z_i, x_i, y_i in zip(self.weights, self.support_vectors, self.supoort_vector_labels):
            result += z_i * y_i *self.kernel(x_i, X)
        return np.sign(result).item()

class Kernel(object):

    @staticmethod
    def linear():
        def f(x,y):
            return np.inner(x,y)
        return f

    @staticmethod
    def gaussian(sigma):
        def f(x,y):
            exponent = -np.sqrt( np.linalg.norm(x-y) **2 / (2* sigma ** 2) )
            return np.exp(exponent)

    def plot(predictor, X, y, grid_size, filename):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),np.linspace(y_min, y_max, grid_size), indexing='ij')
        flatten = lambda m: np.array(m).reshape(-1,)
        result = []
        for (i, j) in itertools.product(range(grid_size), range(grid_size)):
            point = np.array([xx[i, j], yy[i, j]]).reshape(1, 2)
            result.append(predictor.predict(point))
        
        Z = np.array(result).reshape(xx.shape)

        plt.contourf(xx, yy, Z, cmap=cm.Paired, levels=[-0.001, 0.001], extend='both', alpha=0.8)
        plt.scatter(flatten(X[:, 0]), flatten(X[:, 1]),c=flatten(y), cmap=cm.Paired)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.show()


if __name__ == "__main__":

    num_samples=10
    num_features=2
    grid_size=20
    filename="svm.pdf"
    samples = np.matrix(np.random.normal(size=num_samples * num_features).reshape(num_samples, num_features))
    labels = 2 * (samples.sum(axis=1) > 0) - 1.0
    trainer = SVMModel(Kernel.linear(), 0.1)
    predictor = trainer.fit(samples, labels)
    plot(predictor, samples, labels, grid_size, filename)






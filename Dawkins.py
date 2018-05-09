import numpy as np
import tensorflow as tf

tf.reset_default_graph()  

class Worker(object):
    def __init__(self, sess, n_features, n_classes):
        self.sess = sess
        self.X_ = tf.placeholder(tf.float32, shape=(None, n_features))
        self.y_ = tf.placeholder(tf.float32, [None, n_classes])
        self.w_ = tf.placeholder(tf.float32, shape=([n_features, n_classes]))
        
        z1 = tf.matmul(self.X_, self.w_)
        probs = tf.nn.softmax(z1)
        predictions = tf.argmax(probs, 1)
        correct_prediction = tf.equal(predictions, tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
    def run(self, X, y, w):
        """
        for a given set of <X,y> pair and weights, calculate 
        accuracy. in this example, the fitness score is 
        accuracy.
        """
        return self.sess.run(self.accuracy, feed_dict={self.X_:X, self.y_:y, self.w_:w})
    
class Dawkins(object):
    """
    at every iteration (“generation”), a population of 
    parameter vectors (“genotypes”) is perturbed (“mutated”)
    and their objective function value (“fitness”) is 
    evaluated.

    attributes
    ----------
    n_pop : int
        number of individuals in a population
    sigma : float
        constant weighting factor for noise, usually 0.1
    alpha : float
        learning rate, usually 0.001
    n_generations : int
        num of generations
    ls_r : list of floats
        history of fitness scores
    """
    def __init__(self, n_pop=150, n_generations=2000):
        self.n_pop = n_pop
        self.sigma = 0.1
        self.alpha = 0.001
        self.n_generations = n_generations
        self.ls_r = []

    def fit(self, X, y):
        
        self.n_features = X.shape[1]
        self.n_classes = y.shape[1]
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # start off with a random global genotype
            individual = Worker(sess, self.n_features, self.n_classes)
            w = np.random.random_sample((self.n_features, self.n_classes))

            for generation in range(self.n_generations):

                # for each generation, we'll perturb the global genotype with noise
                N = np.random.randn(self.n_pop, self.n_features, self.n_classes)  # noise to be injected
                R = np.zeros(self.n_pop)  # list of fitness scores = accuracy

                # for each individual, try out each genotype
                for j in range(self.n_pop):
                    w_try = w + self.sigma * N[j]  # represents an unique genotype
                    R[j] = individual.run(X, y, w_try)  # corresponding fitness score for the genotype

                # track average fitness of the entire population - if we're doing well, then average fitness should increase
                self.ls_r.append(np.mean(R))
                if generation % (self.n_generations / 10) == 0:
                    print(generation, np.mean(R))

                # update global parameter vector after each generation
                A = (R - np.mean(R)) / np.std(R + 1e-6)

                # reshape to rank 2 matrix, such that rows=individuals, cols=genotypes
                M = N.reshape(self.n_pop, self.n_features * self.n_classes)

                # an individual's genotype will be weighted by its corresponding normalized fitness score
                w = w + self.alpha/(self.n_pop*self.sigma) * np.dot(M.T, A).reshape(4,3)
        
        self.coef_ = w
        
    def predict(self, X, y):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            individual = Worker(sess, self.n_features, self.n_classes)
            res = individual.run(X, y, self.coef_)
            print("using evolved weights, accuracy/fitness: ", res)
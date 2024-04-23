import numpy as np
import scipy
import scipy.stats

class NormalApproximator:
    def fit(X, iter=5, k=1, mu_0=None, var_0=None, debug=False):
        n = len(X)
        v = n

        if mu_0 == None:
            guess_mu = np.random.rand(1)
        else:
            guess_mu = mu_0

        if var_0 == None:
            guess_var = np.random.rand(1)
        else:
            guess_var = var_0

        mean_X = np.mean(X)

        for _ in range(iter):
            mu_n = (k * guess_mu + n * mean_X) / (k + n)
            var_n = (v * guess_var + (n - 1) * np.var(X) + \
                     (k * n * (mean_X - guess_mu)**2 / (k + n))) / (v + n)

            guess_mu = mu_n
            guess_var = (v / 2 * var_n) / (v / 2 - 1)

            if debug:
                print(f"sample fit level: {NormalApproximator.sample_fit_level(X, guess_mu, guess_mu)}")
                print(f"loss: {NormalApproximator.loss(X, guess_mu, guess_mu)}")

        return guess_mu, guess_var
    
    def sample_fit_level(X, guess_mu, guess_var):
        sum = 0

        for x in range(len(X)):
            sum += scipy.stats.norm.pdf(x, loc=guess_mu, scale=np.sqrt(guess_var))

        return sum
    
    def loss(X, guess_mu, guess_var):
        log_loss = 0

        for x in range(len(X)):
            log_loss += np.log(scipy.stats.norm.pdf(
                x, loc=guess_mu, scale=np.sqrt(guess_var)) + 1e-19)

        return log_loss
    
class Normal_MLE:
    def fit(X, mu_0 = None, var_0 = None, iter=5):
        if mu_0 == None:
            mu = np.random.rand(1)
        else:
            mu = mu_0
        
        if var_0 == None:
            var = np.random.rand(1)
        else:
            var = var_0

        for _ in range(iter):
            var = np.sum((X - mu)**2) / len(X)
            mu = np.mean(X)

            pass

        return mu, var
 
def draw_points(mu, var, num_points=100):
    X = np.empty((num_points,))

    for i in range(num_points):
        X[i] = np.random.normal(loc=mu, scale=np.sqrt(var))

    return X


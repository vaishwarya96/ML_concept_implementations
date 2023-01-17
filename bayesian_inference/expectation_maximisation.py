#Based on the code from https://www.ritchievink.com/blog/2019/05/24/algorithm-breakdown-expectation-maximization/

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy as sp

np.random.seed(1)
#For illustration of GMM and expectation maximization, we will consider 2 Gaussians. 
#For this, we draw samples from 2 Gaussians. z_i~Bernoulli(phi)
generative_m = np.array([stats.norm(2,1), stats.norm(5,1.8)])
#z_i is used to determine from which Gaussian distribution we are sampling. Here, a Bernoulli distribution is used with probability of 0.75, which means that 75% of the time, Guassian distribution 1 will be selected and the remaining times, the distribution 0 will be selected.
z_i = stats.bernoulli(0.75).rvs(100)
# Sample a point randomly from each of the Gaussian distribution
x_i = np.array([g.rvs() for g in generative_m[z_i]])

#Plot the probability distributions. In the plot the blue plot gives p(x_i|z_i=0;\theta) and the orange one shows p(x_i|z_i=1;\theta) where \theta are the distribution parameters \mu, \sigma, \phi
x = np.linspace(-5,12,150)
plt.figure(figsize=(16,6))
plt.plot(x, generative_m[0].pdf(x))
plt.plot(x, generative_m[1].pdf(x))
plt.plot(x, generative_m[0].pdf(x)+generative_m[1].pdf(x),lw=1,ls='-.',color='k')
plt.fill_betweenx(generative_m[0].pdf(x), x, alpha=0.1)
plt.fill_betweenx(generative_m[1].pdf(x), x, alpha=0.1)
plt.vlines(x_i, 0, 0.01, color=np.array(['C0', 'C1'])[z_i])
plt.show()


#EM algorithm #1
#As we know, x_i is observed but z_i in not. Otherwise, we could have modelled the joint distribution p(x_i, z_i) = p(x_i|z_i)p(z_i) (i.e fit the Gaussians) by applying log-likelihood estimation. In the EM algorithm, we are going to guess Z and iteratively try to maximize the log-likelihood.
#The log likelihood is given as l(\theta) = \sum_{i=1}^{n} log p(x_i;\theta) = \sum_{i=1}^{n} log \sum_{k=1}^{K}p(x_i|z_k;\theta)p(z_k;\theta). But this is intractable since we do not know z_i. In this, we guess Z and iteratively try to maximize the log-likelihood.
#It is composed of 2 steps: E-Step and M-Step
#E-step- In this, given an initial set of parameters, we determine w_{ij} for every data point: w_{ij}=p(z_i=j|x_i;\theta)
#M-Step- In this we are going to optimize the parameters: \phi_j=1/n\sum_{i=1}^{n}w_{ij}; \mu_j=\frac{\sum_{i=1}^{n}w_{ij}x_i}{\sum_{i=1}^{n}w_{ij}}(weighted mean) and \sigma_j is the weighted variance.

#If we iterate the E-M steps, we hope to converge to maximum likelihood. (The log-likelihood function, is multi-modal so we could get stuck in a local optimum)

class EM:
    def __init__(self, k):
        self.k = k
        self.mu = None
        self.std = np.ones(k)
        self.w_ij = None
        self.phi = np.ones(k) / k

    def expectation_step(self, x):
        for z_i in range(self.k):
            self.w_ij[z_i] = stats.norm(self.mu[z_i], self.std[z_i]).pdf(x) * self.phi[z_i]
	    # normalize zo that marginalizing z would lead to p = 1
        self.w_ij /= self.w_ij.sum(0)

    def maximization_step(self, x):
        self.phi = self.w_ij.mean(1)
        self.std = ((self.w_ij * (x - self.mu[:, None])**2).sum(1) / self.w_ij.sum(1))**0.5
        self.mu = (self.w_ij * x).sum(1) / self.w_ij.sum(1)

    def fit(self, x):
        self.mu = np.random.uniform(x.min(), x.max(), size=self.k)
        self.w_ij = np.zeros((self.k, x.shape[0]))

        last_mu = np.ones(self.k) * np.inf
        while ~np.all(np.isclose(self.mu, last_mu)):
            last_mu = self.mu
            self.expectation_step(x)
            self.maximization_step(x)

m = EM(2)
m.fit(x_i)

    
m = EM(2)
m.fit(x_i)

fitted_m = [stats.norm(mu, std) for mu, std in zip(m.mu, m.std)]

plt.figure(figsize=(16, 6))
plt.vlines(x_i, 0, 0.01, color=np.array(['C0', 'C1'])[z_i])
plt.plot(x, fitted_m[0].pdf(x))
plt.plot(x, fitted_m[1].pdf(x))
plt.plot(x, generative_m[0].pdf(x), color='black', lw=1, ls='-.')
plt.plot(x, generative_m[1].pdf(x), color='black', lw=1, ls='-.')
plt.show()



#Based on https://zhiyzuo.github.io/EM/
class GMM(object):
    def __init__(self, X, k=2):
        # dimension
        X = np.asarray(X)
        self.m, self.n = X.shape
        self.data = X.copy()
        # number of mixtures
        self.k = k
        
    def _init(self):
        # init mixture means/sigmas
        self.mean_arr = np.asmatrix(np.random.random((self.k, self.n)))
        self.sigma_arr = np.array([np.asmatrix(np.identity(self.n)) for i in range(self.k)])
        self.phi = np.ones(self.k)/self.k
        self.w = np.asmatrix(np.empty((self.m, self.k), dtype=float))
        print(self.mean_arr)
        print(self.sigma_arr)
        print(self.phi)
    
    def fit(self, tol=1e-4):
        self._init()
        num_iters = 0
        ll = 1
        previous_ll = 0
        while(ll-previous_ll > tol):
            previous_ll = self.loglikelihood()
            self._fit()
            num_iters += 1
            ll = self.loglikelihood()
            print('Iteration %d: log-likelihood is %.6f'%(num_iters, ll))
        print('Terminate at %d-th iteration:log-likelihood is %.6f'%(num_iters, ll))
    
    def loglikelihood(self):
        ll = 0
        for i in range(self.m):
            tmp = 0
            for j in range(self.k):
                #print(self.sigma_arr[j])
                tmp += sp.stats.multivariate_normal.pdf(self.data[i, :], 
                                                        self.mean_arr[j, :].A1, 
                                                        self.sigma_arr[j, :]) *\
                       self.phi[j]
            ll += np.log(tmp) 
        return ll
    
    def _fit(self):
        self.e_step()
        self.m_step()
        
    def e_step(self):
        # calculate w_j^{(i)}
        for i in range(self.m):
            den = 0
            for j in range(self.k):
                num = sp.stats.multivariate_normal.pdf(self.data[i, :], 
                                                       self.mean_arr[j].A1, 
                                                       self.sigma_arr[j]) *\
                      self.phi[j]
                den += num
                self.w[i, j] = num
            self.w[i, :] /= den
            assert self.w[i, :].sum() - 1 < 1e-4
            
    def m_step(self):
        for j in range(self.k):
            const = self.w[:, j].sum()
            self.phi[j] = 1/self.m * const
            _mu_j = np.zeros(self.n)
            _sigma_j = np.zeros((self.n, self.n))
            for i in range(self.m):
                _mu_j += (self.data[i, :] * self.w[i, j])
                _sigma_j += self.w[i, j] * ((self.data[i, :] - self.mean_arr[j, :]).T * (self.data[i, :] - self.mean_arr[j, :]))
                #print((self.data[i, :] - self.mean_arr[j, :]).T * (self.data[i, :] - self.mean_arr[j, :]))
            self.mean_arr[j] = _mu_j / const
            self.sigma_arr[j] = _sigma_j / const
        #print(self.sigma_arr)


X = np.random.multivariate_normal([0, 3], [[0.5, 0], [0, 0.8]], 20)
X = np.vstack((X, np.random.multivariate_normal([20, 10], np.identity(2), 50)))
X.shape

gmm = GMM(X)
gmm.fit()

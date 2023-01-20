# Taken from this post: https://www.ritchievink.com/blog/2019/09/16/variational-inference-from-scratch/

import numpy as np
from torch import nn
from sklearn import datasets
import matplotlib.pyplot as plt
import torch

#Generate data with noise dependent on X
w0 = 0.125
b0 = 5.
x_range = [-20, 60]

def load_dataset(n=150, n_tst=150):
    np.random.seed(43)

    def s(x):
        g = (x - x_range[0])/(x_range[1] - x_range[0])
        return 3 * (0.25 + g**2)


    x = (x_range[1] - x_range[0]) * np.random.rand(n) + x_range[0]
    eps = np.random.randn(n) * s(x)
    y = (w0 * x * (1. + np.sin(x)) + b0) + eps
    y = (y - y.mean()) / y.std()
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    return y[:, None], x[:, None]

y, x = load_dataset()

plt.figure(figsize=(16,6))
plt.scatter(x,y)


# Maximum likelihood estimate

# We will first model a neural network with MLE. We assume a Gaussian Likelihood.
#y ~ Gaussian with means g_\theta(x) and variance \sigma^2
#\theta_MLE = argmax_\theta \prod_i=1^n P(y_i|\theta)

X = torch.tensor(x, dtype=torch.float)
Y = torch.tensor(y, dtype=torch.float)

class MaximumLikelihood(nn.Module):
    def __init__(self):
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(1,20),
            nn.ReLU(),
            nn.Linear(20,1)
        )

    def forward(self, x):
        return self.out(x)

epochs = 200
m = MaximumLikelihood()
optim = torch.optim.Adam(m.parameters(), lr=0.01)

#Fit model using point estimates
for epoch in range(epochs):
    optim.zero_grad()
    y_pred = m(X)
    loss = (0.5 * (y_pred - Y)**2).mean()
    loss.backward()
    optim.step()

plt.plot(x, y_pred.detach().numpy())
plt.show()

#Here, the model has predicted point estimates using MLE, but we have no estimate about the noise in our predictions.

#Variational Regression
#Consider a model where we would like to model p(y|x) proportional to to p(x|y)p(y). In VI, we can't model the true posterior. So we approximate it with another distribution q_\theta(y), where \theta are the variational paramaters. This distribution is called the variational distribution.

#If we choose a factorized (diagonal) Gaussian variational distribution, Qθ(y) becomes Qθ(μ,diag(σ^2)). Note that we are now working with an 1D case and that this factorization doesn’t mean much right now. We want this distribution to be conditioned to x, therefore we define a function gθ:x↦μ,σ. The function gθ will be a neural network that predicts the variational parameters. The total model can thus be described as:
# P(y)=N(0,1)
# Q(y|x)=N(g_θ(x)_μ,diag(g_θ(x)_σ^2)))
#We set a unit Gaussian prior P(y)

#Optimization problem
#In this, we have defined y|x as the variational distribution but it can used for any latent variable Z. So from now on we will use the convention of z. 

#We will now implement VI with reparameterization trick

class VI(nn.Module):
    def __init__(self):
        super().__init__()

        self.q_mu = nn.Sequential(
            nn.Linear(1, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

        self.q_log_var = nn.Sequential(
            nn.Linear(1, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def reparameterize(self, mu, log_var):   #z = mu+sigma*epsilon
        #std can't be negative, so we use log variance
        sigma = torch.exp(0.5 * log_var) + 1e-5
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def forward(self, x):
        mu = self.q_mu(x)
        log_var = self.q_log_var(x)
        return self.reparameterize(mu, log_var), mu, log_var


#Log likelihood of a Gaussian distribution

def ll_gaussian(y, mu, log_var):
    sigma = torch.exp(0.5 * log_var)
    return -0.5 * torch.log(2 * np.pi * sigma**2) - (1/ (2 * sigma**2)) * (y-mu)**2

def elbo(y_pred, y, mu, log_var):
    # Likelihood of observing y, given variational mu and sigma
    likelihood = ll_gaussian(y, mu, log_var)

    #Prior probability of y_pred
    log_prior = ll_gaussian(y_pred, 0, torch.log(torch.tensor(1.)))

    #Variational probability of y_pred
    log_p_q = ll_gaussian(y_pred, mu, log_var)

    #By taking the mean we approximate the expectation
    return (likelihood + log_prior - log_p_q).mean()

def det_loss(y_pred, y, mu, log_var):
    return -elbo(y_pred, y, mu, log_var)

epochs = 1500

m = VI()
optim = torch.optim.Adam(m.parameters(), lr=0.005)

for epoch in range(epochs):
    optim.zero_grad()
    y_pred, mu, log_var = m(X)
    loss = det_loss(y_pred, Y, mu, log_var)
    loss.backward()
    optim.step()


# draw samples from Q(theta)
with torch.no_grad():
    y_pred = torch.cat([m(X)[0] for _ in range(1000)], dim=1)
    
# Get some quantiles
q1, mu, q2 = np.quantile(y_pred, [0.05, 0.5, 0.95], axis=1)

plt.figure(figsize=(16, 6))
plt.scatter(X, Y)
plt.plot(X, mu)
plt.fill_between(X.flatten(), q1, q2, alpha=0.2)
plt.show()

    
#Analytical KL divergence and reconstruction loss
#Above we have implemented ELBO by sampling from the variational posterior. It turns out that for the KL-divergence term, this isn’t necessary as there is an analytical solution. For the Gaussian case, Diederik P. Kingma and Max Welling (2013. Auto-encoding variational bayes) included the solution in Appendix B.
# For the likelihood term, we did implement Guassian log likelihood, this term can also be replaced with a similar loss functions. For Gaussian likelihood we can use squared mean error loss, for Bernoulli likelihood we could use binary cross entropy etc. If we do that for the earlier defined model, we can replace the loss function as defined below:

def det_loss(y, y_pred, mu, log_var):    
    reconstruction_error = (0.5 * (y - y_pred)**2).sum()
    kl_divergence = (-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp()))

    return (reconstruction_error + kl_divergence).sum()

# Aleatoric and epistemic uncertainty
#The above was an example of modelling aleatoric uncertainty
#Now we will see how we can model epistemic uncertainty
class LinearVariational(nn.Module):
    def __init__(self, in_features, out_features, parent, n_batches, bias=True):
        super().__init__()
        
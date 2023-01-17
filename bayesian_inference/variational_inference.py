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

    

#https://www.ritchievink.com/blog/2019/06/10/bayesian-inference-how-we-are-able-to-chase-the-posterior/

#Using Bayes formula, we would like to find the model parameters, given the data. However, the marginalisation term usually uses an integral which is generally intractable. At even moderately high dimensions of θ the amount numerical operations explode. 

#Let's start with a simple example

#Assume that we have observed two data points; D={195,182}. Both are observed lengths in cm of men in a basketball competition.

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

lengths = np.array([195, 182])

#Likelihood function

#Let us assume that the distribution of the true weights follow a Gaussian distribution. A Gaussian is parameterized with a mean μ and variance σ2. For a reasonable domain of these parameters θ={μ,σ} we can compute the likelihood P(D|θ)=P(D|μ,σ).

#This is the computation domain
# lets create a grid of our two parameters
mu = np.linspace(150, 250)
sigma = np.linspace(0, 15)[::-1]

mm, ss = np.meshgrid(mu, sigma)  # just broadcasted parameters

#Likelihood
likelihood = stats.norm(mm, ss).pdf(lengths[0]) * stats.norm(mm, ss).pdf(lengths[1])  #Remember, from Andrew Ng's class, L(theta) is the product of probability of data.
aspect = mm.max() / ss.max() / 3
extent = [mm.min(), mm.max(), ss.min(), ss.max()]

plt.imshow(likelihood, cmap='Reds', aspect=aspect, extent=extent)
plt.xlabel(r'$\mu$')
plt.ylabel(r'$\sigma$')
plt.show()

#As we can see, the likelihood function represents the most likely parameters. If we would infer the most likely parameters θ based on only the likelihood we would choose the darkest red spots in the plot. By eyeballing it, I would say that μ=190 and σ=5.

#Prior distribution
#Besides the likelihood, Bayes' rule allows us to also include our prior belief in estimating the parameters. Assume that the mean follows a Gaussian distribution and variance follows a Cauchy distribution.

prior = stats.norm(200, 15).pdf(mm) * stats.cauchy(0, 10).pdf(ss)

plt.imshow(prior, cmap='Greens', aspect=aspect, extent=extent)
plt.xlabel(r'$\mu$')
plt.ylabel(r'$\sigma$')
plt.show()

#Posterior distribution
#As we now have a simple model, not more than two dimensions, and a reasonable idea in which domain we need to search, we can compute the posterior directly by applying Bayes' rule.
unnormalized_posterior = prior * likelihood
posterior = unnormalized_posterior / np.nan_to_num(unnormalized_posterior).sum()
plt.imshow(posterior, cmap='Blues', aspect=aspect, extent=extent)
plt.xlabel(r'$\mu$')
plt.ylabel(r'$\sigma$')
plt.show()

#This was easy. Because we weren’t cursed by high dimensions. Increase the dimensions, or define a more complex model, and the calculation of P(D) becomes intractable.
# Even calculating p(D) is not really necessary because it only acts as a normalizer. We can directly use unnormalized posterior to make predictions. However, it is still expensive.

#There are two methods to tackle this: MCMC and Variational Inference

#MCMC 
'''
This is done by exploring θ space by taking a random walk and computing the joint probability P(θ,D) and saving the parameter sample of θi

according to the following probability:

P(acceptance)=min(1,(P(D|θ∗)P(θ∗))/(P(D|θ)P(θ)))

Where θ= current state, θ∗= proposal state.

The proposals that were accepted are samples from the actual posterior distribution. This is of course very powerful, as we are able to directly sample from, and therefore approximate, the real posterior!

Now we can see the relation with the name of the algorithm.

    Markov Chain: A chain of events, where every new event depends only on the current state; Acceptance probability.
    Monte Carlo: Doing something at random; Random walk through θ

    space.
'''

#Let us now implement MCMC with PyMC3
import pymc3 as pm

with pm.Model():
    # priors
    mu = pm.Normal('mu', mu=200, sd=15)
    sigma = pm.HalfCauchy('sigma', 10)
    
    # likelihood
    observed = pm.Normal('observed', mu=mu, sd=sigma, observed=lengths)
    # sample
    trace = pm.sample(draws=10000, chains=1)

fig, axes = plt.subplots(2, sharex=True, sharey=True, figsize = (16, 6))
axes[0].imshow(posterior, cmap='Blues', extent=extent, aspect=1)
axes[0].set_ylabel('$\sigma$')
axes[1].scatter(trace['mu'], trace['sigma'], alpha=0.01)
axes[1].set_ylabel('$\sigma$')
axes[0].set_title('True posterior')
axes[1].set_title('Sampled $\\theta$')
plt.xlabel('$\mu$')
plt.xlim(150, mm.max())
plt.ylim(0, ss.max())
plt.show()


#Variational Inference
'''
If we go into the realm of deep learning with large datasets and potentially millions of parameters
θ

, sampling with MCMC is often too slow. For this kind of problems, we rely on a technique called Variational Inference. Instead of computing the distribution, or approximating the real posterior by sampling from it, we choose an approximated posterior distribution and try to make it resemble the real posterior as close as possible.

The drawback of this method is that our approximation of the posterior can be really off.

The benefit is that it is an optimization problem with flowing gradients, which in the current world of autograd isn’t a problem at all!
'''

import pyro
import pyro.optim
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
import torch
import torch.distributions.constraints as constraints


def model():
    # priors
    mu = pyro.sample('mu', dist.Normal(loc=torch.tensor(200.), 
                                       scale=torch.tensor(15.)))
    sigma = pyro.sample('sigma', dist.HalfCauchy(scale=torch.tensor(10.)))
    
    # likelihood
    with pyro.plate('plate', size=2):
        pyro.sample(f'obs', dist.Normal(loc=mu, scale=sigma), 
                    obs=torch.tensor([195., 185.]))

def guide():
    # variational parameters
    var_mu = pyro.param('var_mu', torch.tensor(180.))
    var_mu_sig = pyro.param('var_mu_sig', torch.tensor(5.),
                             constraint=constraints.positive)
    var_sig = pyro.param('var_sig', torch.tensor(5.))

    # factorized distribution
    pyro.sample('mu', dist.Normal(loc=var_mu, scale=var_mu_sig))
    pyro.sample('sigma', dist.Chi2(var_sig))

pyro.clear_param_store()
pyro.enable_validation(True)

svi = SVI(model, guide, 
          optim=pyro.optim.ClippedAdam({"lr":0.01}), 
          loss=Trace_ELBO())

# do gradient steps
c = 0
for step in range(5000):
    c += 1
    loss = svi.step()
    if step % 100 == 0:
        print("[iteration {:>4}] loss: {:.4f}".format(c, loss))

sigma = dist.Chi2(pyro.param('var_sig')).sample((10000,)).numpy()
mu = dist.Normal(pyro.param('var_mu'), pyro.param('var_mu_sig')).sample((10000,)).numpy()

fig, axes = plt.subplots(2, sharex=True, sharey=True, figsize = (16, 6))
axes[0].imshow(posterior, cmap='Blues', extent=extent, aspect=1)
axes[0].set_ylabel('$\sigma$')
axes[1].scatter(mu, sigma, alpha=0.1)
axes[1].set_ylabel('$\sigma$')
axes[0].set_title('True posterior $P(\\theta|D)$')
axes[1].set_title('Approximated posterior $Q(\\theta)$')
plt.xlabel('$\mu$')
plt.xlim(150, mm.max())
plt.ylim(0, ss.max())

plt.show()
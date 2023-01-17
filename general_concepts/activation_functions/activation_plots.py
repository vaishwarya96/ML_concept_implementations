import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


x_range = np.arange(-1,1,0.1)
y_range = np.arange(0,1,0.1)
x_tensor = torch.Tensor(x_range)

#Sigmoid activation
sigma = nn.Sigmoid()(x_tensor)
gradient = sigma*(1-sigma)

#Tanh activation
sigma = nn.Tanh()(x_tensor)
gradient = 1-sigma**2

#Relu activation
sigma = nn.ReLU()(x_tensor)
gradient = torch.zeros(sigma.shape)
gradient[sigma>0] = 1

#Leaky ReLU
slope=0.01
sigma = nn.LeakyReLU(negative_slope=slope)(x_tensor)
gradient = torch.ones(sigma.shape)
gradient[x_tensor<0] = slope


#ELU
alpha=1
sigma = nn.ELU(alpha)(x_tensor)
gradient = torch.ones(sigma.shape)
gradient[x_tensor<0] = sigma[x_tensor<0] + alpha

#SiLU activation
sigma = nn.SiLU()(x_tensor)
gradient = sigma*(1+x_tensor*(1-sigma))



plt.plot(x_range, sigma, label='Sigma')
plt.plot(x_range, gradient, label='Gradient')
plt.grid()
plt.ylabel('sigma(x)')
plt.xlabel('x')
plt.legend()
plt.show()

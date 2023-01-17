#This is an example of implementation for Newton's method for optimization. The assumption made here is that if f(x) is the function to be optimized, f(x) is twice differentiable and not equal to 0.

import numpy as np
import math
import matplotlib.pyplot as plt

#An example of a function for finding the local maxima
def f(x):
    return x**3-6*x**2+4*x+2

def fprime(x):
    return 3*x**2-12*x+4

def fsecondprime(x):
    return 6*x - 12

def quadratic_approx(x, x0, f, fprime, fsecond):
    return f(x0)+fprime(x0)*(x-x0)+0.5*fsecond(x0)*(x-x0)**2

def newton(x0, fprime, fsecond, maxiter=100, eps=0.0001):
    x=x0
    for i in range(maxiter):
        xnew=x-(fprime(x)/fsecond(x))
        if xnew-x<eps:
            return xnew
            print('converged')
            break
        x = xnew
    return x


#To optimize using newton's method to find the optimum value (maximum in this case), we should start by approximating our curve in a random starting point with a second order Taylor expansion, and compute the maximum of the extension.


#First plot the function
x = np.linspace(-1,1)
fig, ax = plt.subplots()
ax.plot(x, f(x), label='target')
ax.grid()

ax.plot(x, quadratic_approx(x, 0, f, fprime, fsecondprime), color='red', label='quadratic approximation')
ax.set_ylim([-2,3])
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
x_star=newton(0, fprime, fsecondprime)
ax.axvline(x = x_star, color='green')
plt.legend()

plt.show()


# Now an example on finding local minima. In this case, determining the local maxima or minima depends on where the initialization is done. If initial point is close to the maxima, then the algorithm finds the local maximia. If close to minima, it finds the local minima.

def f(x):
    return np.sin(x)

def fprime(x):
    return np.cos(x)

def fsecondprime(x):
    return -np.sin(x)

def quadratic_approx(x, x0, f, fprime, fsecond):
    return f(x0)+fprime(x0)*(x-x0)+0.5*fsecond(x0)*(x-x0)**2

def newton(x0, fprime, fsecond, maxiter=100, eps=0.0001):
    x=x0
    for i in range(maxiter):
        xnew=x-(fprime(x)/(fsecond(x)+1e-20))
        if xnew-x<eps:
            return xnew
            print('converged')
            break
        x = xnew
    return x


#To optimize using newton's method to find the optimum value (maximum in this case), we should start by approximating our curve in a random starting point with a second order Taylor expansion, and compute the maximum of the extension.


#First plot the function
x = np.linspace(-5,5)
fig, ax = plt.subplots()
ax.plot(x, f(x), label='target')
ax.grid()

ax.plot(x, quadratic_approx(x, -1, f, fprime, fsecondprime), color='red', label='quadratic approximation')
ax.set_ylim([-2,3])
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
x_star=newton(-1, fprime, fsecondprime)
ax.axvline(x = x_star, color='green')
plt.legend()

plt.show()

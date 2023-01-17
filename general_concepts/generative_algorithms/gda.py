import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal
from sklearn.metrics import accuracy_score

rng = default_rng(42)

def get_MLE(x_train, y_train , num_classes=2):
    '''
    This function is used to calculate the \mu, \Sigma and the \phi values
    '''
    y_train = y_train.squeeze(1)
    phi = np.zeros((num_classes, 1))
    mu = np.zeros((num_classes, x_train.shape[1]))
    sigma = np.zeros((num_classes, x_train.shape[1], x_train.shape[1]))

    num_samples = x_train.shape[0]
    for i in range(num_classes):
        idx = (y_train == i)
        phi[i] = np.sum(idx)/num_samples
        mu[i] = np.mean(x_train[idx,:], axis=0)
        sigma[i] = np.cov(x_train[idx,:], rowvar=0)

    return phi, mu, sigma


def predict(x_test, phi, mu, sigma):

    num_classes = phi.shape[0]
    pred_prob = np.zeros((x_test.shape[0],num_classes))
    for i in range(num_classes):
        likelihood = multivariate_normal.pdf(x_test, mean=mu[i], cov=sigma[i])   #Likelihood is obtained from the PDF of multivariate normal distribution
        pred_prob[:,i] = likelihood * phi[i]   #Here, phi is the prior
        
    pred_prob /= np.sum(pred_prob, axis=1, keepdims=True)   #Divide by the marginalisation term
    
    return pred_prob

def test(pred_prob, y_test):
    pred_label = np.argmax(pred_prob, axis=1)
    accuracy = accuracy_score(pred_label, y_test)
    return accuracy





N_POINTS = 500
SCALE = 0.05
N_CLASSES = 3

features = np.r_[rng.normal((0.4,0.5), SCALE, (N_POINTS, 2)), rng.normal((0.6, 0.5), SCALE, (N_POINTS, 2)), rng.normal((0.5, 0.3), SCALE, (N_POINTS, 2))]
labels = np.r_[np.zeros((N_POINTS, 1)), np.ones((N_POINTS, 1)), np.ones((N_POINTS, 1))*2]
X_train, X_test, y_train, y_test = train_test_split(features, labels, stratify=labels)

phi, mu, sigma = get_MLE(X_train, y_train, N_CLASSES)
pred_prob = predict(X_test, phi, mu, sigma)
accuracy = test(pred_prob, y_test)
print("Accuracy is: %f" %(accuracy))


xmin, xmax = 0,1
ymin, ymax = 0,1
xmin, xmax, ymin, ymax = int(xmin)-0.1, int(xmax)+0.1, int(ymin)-0.1, int(ymax)+0.1
xx, yy = np.meshgrid(np.arange(xmin, xmax, 0.05), np.arange(ymin, ymax, 0.05))
eval_dataset = np.c_[xx.ravel(), yy.ravel()]

f, axarr = plt.subplots()
pred_prob = predict(eval_dataset, phi, mu, sigma)
axarr.scatter(X_train[:, 0], X_train[:, 1], c= y_train, alpha=0.4)
prediction = np.max(pred_prob, axis=1).reshape(xx.shape)
cs1 = axarr.contourf(xx, yy, prediction, alpha=0.5)
f.colorbar(cs1)
plt.show()
#https://towardsdatascience.com/svm-implementation-from-scratch-python-2db2fc52e5c2
#https://programmathically.com/understanding-hinge-loss-and-the-svm-cost-function/

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
from sklearn.utils import shuffle
import statsmodels.api as sm

#Feature selection
def remove_correlated_features(X):
    corr_threshold = 0.9
    corr = X.corr()
    drop_columns = np.full(corr.shape[0], False, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if corr.iloc[i, j] >= corr_threshold:
                drop_columns[j] = True
    columns_dropped = X.columns[drop_columns]
    X.drop(columns_dropped, axis=1, inplace=True)
    return columns_dropped

def remove_less_significant_features(X, Y):
    sl = 0.05
    regression_ols = None
    columns_dropped = np.array([])
    for itr in range(0, len(X.columns)):
        regression_ols = sm.OLS(Y, X).fit()
        max_col = regression_ols.pvalues.idxmax()
        max_val = regression_ols.pvalues.max()
        if max_val > sl:
            X.drop(max_col, axis='columns', inplace=True)
            columns_dropped = np.append(columns_dropped, [max_col])
        else:
            break
    regression_ols.summary()
    return columns_dropped

#Model training
def compute_cost(W, X, Y):
    
    #Hinge loss
    N = X.shape[0] 
    print(W.shape, X.shape, Y.shape)
    distances = 1 - Y * (np.dot(W, X))   #Assuming Y is +/-1
    distances[distances<0] = 0  #When distance<0, it means that the point lies well beyond the margin and so the loss is 0
    hinge_loss = reg_strength * (np.sum(distances) / N)

    cost = 0.5 * np.dot(W*W) + hinge_loss
    return cost

def calculate_cost_gradient(W, X_batch, Y_batch):
    # if only one example is passed as in the case of SGD
    if isinstance(Y_batch, np.float64):
        #Y_batch = np.array([Y_batch])
        Y_batch = Y_batch.tolist()
        X_batch = np.array([X_batch]).reshape(1, X_batch.shape[0])
    
    distance = 1 - (Y_batch * np.dot(X_batch, W))
    dw = np.zeros(len(W))

    distance = [distance]

    for idx, d in enumerate(distance):
        if max(0,d) == 0:
            di = W   
        else:
            di = W - (reg_strength * Y_batch * X_batch)
        
        dw += di

    dw = dw/1

    return dw




def sgd(features, outputs):
    max_epochs = 2000
    weights = np.zeros(features.shape[1])
    total_weights = np.zeros(features.shape)

    for epoch in range(1, max_epochs):
        X,Y = shuffle(features, outputs)
        for ind, x in enumerate(X):
            ascent = calculate_cost_gradient(weights,x,Y[ind])
            weights = weights - (learning_rate * ascent)
            total_weights[ind,:] = weights

        '''
        # stoppage criterion, convergence check 
        nth = 0
        prev_cost = float("inf")
        cost_threshold = 0.01 # in percent

        if epoch == 2 ** nth or epoch == max_epochs - 1:
            cost = compute_cost(weights, features, outputs)
            print("Epoch is:{} and Cost is: {}".format(epoch, cost))
            # stoppage criterion
            if abs(prev_cost - cost) < cost_threshold * prev_cost:
                return weights
            prev_cost = cost
            nth += 1
        '''

    return weights    

def init():

    #Load the dataset
    data = pd.read_csv('./data.csv')
    diagnosis_map = {'M':1, 'B':-1}
    data['diagnosis'] = data['diagnosis'].map(diagnosis_map)
    data.drop(data.columns[[-1, 0]], axis=1, inplace=True)

    Y = data.loc[:, 'diagnosis']
    X = data.iloc[:, 1:]
    #Normalise the data
    X_normalised = MinMaxScaler().fit_transform(X.values)
    X = pd.DataFrame(X_normalised)

    remove_correlated_features(X)
    remove_less_significant_features(X, Y)
    X.insert(loc=len(X.columns), column='intercept', value=1)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    #Train the network
    print("Training started!!!")
    W = sgd(X_train.to_numpy(), y_train.to_numpy())
    print("Training completed")
    print("Weights are: {}".format(W))

    #Testing the model on test set
    y_test_predicted = np.array([])
    for i in range(X_test.shape[0]):
        yp = np.sign(np.dot(W, X_test.to_numpy()[i])) #model
        y_test_predicted = np.append(y_test_predicted, yp)
        
    print("accuracy on test dataset: {}".format(accuracy_score(y_test.to_numpy(), y_test_predicted)))
    print("recall on test dataset: {}".format(recall_score(y_test.to_numpy(), y_test_predicted)))
    print("precision on test dataset: {}".format(recall_score(y_test.to_numpy(), y_test_predicted)))

  
    



if __name__ == '__main__':

    reg_strength = 10000
    learning_rate = 0.00001
    init()
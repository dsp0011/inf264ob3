
from sklearn.model_selection import train_test_split
from sklearn import model_selection

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import dask.dataframe as dd


from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import pprint

def get_project_data():
    ### Generates lines to skip, because the dataset is too large for testing hyperparameters
    li = random.sample(range(1,70000), 20000) # 20k unique integers, 10 for developing models, 10k for final model assesment
    # print("li:")
    # print(li)
    li = [1, 20000, 40000, 60000]
    r = [i for i in range(max(li)) if i not in li]
    X = dd.read_csv('handwritten_digits_images.csv')
    y = dd.read_csv('handwritten_digits_labels.csv')

    X = X.compute()
    y = y.compute()
    
    print("y")
    print(y)

    #No shuffling is needed 
    X_train = X[0:10000]
    y_train = y[0:10000]

    X_test = X[10000:]
    y_test = y[10000:]

    print("y_test")

    print(y_test)

    print("y_train")

    print(y_train)


    return (X_train, y_train, X_test, y_test)


#           EVALUATION MEASUREMENTS           

def get_accuracy(y_pred, y_test, model_name):
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy
    # print(model_name + " accuracy: ", accuracy)

def get_cross_validation_score(model, X, y, model_name):
    scoring = ['recall_macro', 'precision_macro', 'accuracy']
    scores = cross_validate(model, X, y, cv=5, scoring=scoring)
    print(scores.keys())
    for key in scores.keys():
        print(model_name, key," : %0.4f (+/- %0.4f)" % (scores[key].mean(), scores[key].std() * 2))
    
    # print(model_name, " accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    # print("wow", (scores["test_recall_macro"].mean(), scores["test_precision_macro"].mean()))
    return (scores["test_recall_macro"].mean(), scores["test_precision_macro"].mean(), scores["test_accuracy"].mean())

def get_confusion_matrix(y_pred, y_test, model_name):
    print(model_name)
    print(confusion_matrix(y_test, y_pred))

#           TESTING HYPERPARAMETERS          

def test_KNN_parameters(X, y, model_name):
    #test nearest neighbour
    accuracy_list = []
    recall_list = []
    precision_list = []
    for i in range(1, 10):
        accuracy, recall, precision = KNN(X, y,i)
        accuracy_list.append(accuracy)
        recall_list.append(recall)
        precision_list.append(precision)
        print("knn = ", i, " accuracy =", accuracy, " recall =", recall, " precision =", precision)

    fig, ax = plt.subplots()
    plt.plot(accuracy_list, c="blue", label="Accuracy")
    plt.plot(recall_list, c="red", label="Recall")
    plt.plot(precision_list, c="yellow", label="Precision")
    ax.legend()
    plt.xticks(rotation=65)
    plt.show()

def test_support_vector_parameters(X, y, model_name):
    accuracy_list = []
    recall_list = []
    precision_list = []
    for i in range(1,10):
    # for i in np.linspace(0.1, 1.0, 10):
        accuracy, recall, precision = support_vector(X, y,"sigmoid",i)
        accuracy_list.append(accuracy)
        recall_list.append(recall)
        precision_list.append(precision)
        print("knn = ", i, " accuracy =", accuracy, " recall =", recall, " precision =", precision)

    fig, ax = plt.subplots()
    plt.plot(accuracy_list, c="blue", label="Accuracy")
    plt.plot(recall_list, c="red", label="Recall")
    plt.plot(precision_list, c="yellow", label="Precision")
    ax.legend()
    plt.xticks(rotation=65)
    plt.show()

def test_MLP_parameters(X, y, model_name):
    accuracy_list = []
    recall_list = []
    precision_list = []
    for i in np.linspace(50, 200, 8):
        accuracy, recall, precision = neural_network(X, y, int(i))
        accuracy_list.append(accuracy)
        recall_list.append(recall)
        precision_list.append(precision)
        print("knn = ", i, " accuracy =", accuracy, " recall =", recall, " precision =", precision)

    fig, ax = plt.subplots()
    plt.plot(accuracy_list, c="blue", label="Accuracy")
    plt.plot(recall_list, c="red", label="Recall")
    plt.plot(precision_list, c="yellow", label="Precision")
    ax.legend()
    ax.set_xticklabels(np.linspace(50, 200, 8))
    plt.xticks(rotation=65)
    plt.show()

#           IMPLEMENTATION  
def KNN(X, y, n_neighbors):
    neigh = KNeighborsClassifier(n_neighbors = n_neighbors, weights="distance")
    neigh.fit(X, y)
    # y_pred = neigh.predict(X_test)
    # accuracy = get_accuracy(y_pred, y_test, ("KNN, n = " + str(n_neighbors)))
    recall, precision, accuracy = get_cross_validation_score(neigh, X, y, ("KNN, n = " + str(n_neighbors)))
    return (accuracy, recall, precision)


def support_vector(X_train, y_train, kernel, degree):
    if kernel == "poly":
        sv = SVC(kernel=kernel, degree=degree)
    else:
        print("degreeL: ", degree)
        sv = SVC(kernel=kernel, C= degree)

    sv.fit(X_train, y_train)
    # y_pred = sv.predict(X_test)

    # get_accuracy(y_pred, y_test, "support vector, kernel: " + kernel)
    # get_cross_validation_score(sv, X_train, y_train, "support vector, kernel: " + kernel)
    # get_confusion_matrix(y_pred, y_test, "support vector, kernel: " + kernel)

    recall, precision, accuracy = get_cross_validation_score(sv, X_train, y_train, ("SV sigmoid, c = " + str(degree)))
    return (accuracy, recall, precision)

def neural_network(X_train, y_train, max_iter):

    mlp = MLPClassifier(activation="relu", solver="adam", max_iter=max_iter)
    mlp.fit(X_train, y_train)
    # y_pred = mlp.predict(X_test) ,

    recall, precision, accuracy = get_cross_validation_score(mlp, X_train, y_train, ("MLP, n = " + str(max_iter)))
    return (accuracy, recall, precision)
    # get_accuracy(y_pred, y_test, "neural network")

def test_final_model(X_train, y_train, X_test, y_test):
    sv = SVC(kernel="rbf", degree=3)
    sv.fit(X_train, y_train)

    y_pred = sv.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Final accuracy: ", acc) 



def show_image(img):
    plt.imshow(img, cmap="Greys")
    plt.show()

def main():
    X_train, y_train, X_test, y_test = get_project_data()

    # support_vector(X_train, y_train.values.ravel(), "poly", 5)
    # neural_network(X_train, y_train.values.ravel())
    # decision_tree(X_train, y_train.values.ravel())
    # KNN(X_train, y_train.values.ravel(), 3)

    # test_KNN_parameters(X_train, y_train.values.ravel(), "wow")
    # test_support_vector_parameters(X_train, y_train.values.ravel(), "wow")
    # test_MLP_parameters(X_train, y_train.values.ravel(), "wow")

    test_final_model(X_train, y_train.values.ravel(), X_test, y_test.values.ravel())

if __name__ == "__main__":
    main()
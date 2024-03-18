import matplotlib.pyplot as plt
import DataGenerator as dataGen
import numpy as np
import scikitplot as skplt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

def draw_plt(data, indexes, title):
    plt.hist(data[indexes, 0], bins='auto', alpha=0.7)
    plt.hist(data[~indexes, 0], bins='auto', alpha=0.7)
    plt.title(title)
    plt.show()
    
def calculate_accurancy(pred, y):
    return sum(pred == y) / len(y)

def calculate_sensitivity_specificity(pred, y):
    TP_test = sum((pred == 1) & (y == 1))
    FN_test = sum((pred == 0) & (y == 1))
    TN_test = sum((pred == 0) & (y == 0))
    FP_test = sum((pred == 1) & (y == 0))

    sensitivity = TP_test / (TP_test + FN_test)
    specificity = TN_test / (TN_test + FP_test)
    return sensitivity, specificity

def predict(x, y, class0, class1):
    _ = plt.scatter(class0[:, 0], class0[:, 1], marker=".", alpha=0.7)
    _ = plt.scatter(class1[:, 0], class1[:, 1], marker=".", alpha=0.7)
    plt.show()

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=42)

    Nvar = 17
    clf = LogisticRegression(random_state=Nvar, solver='saga').fit(xtrain, ytrain)

    Pred_test = clf.predict(xtest)
    Pred_test_proba = clf.predict_proba(xtest)
    title_test = "res1"
    draw_plt(Pred_test_proba, ytest, title_test)

    Pred_train = clf.predict(xtrain)
    Pred_train_proba = clf.predict_proba(xtrain)
    title_train = "res2"
    draw_plt(Pred_train_proba, ytrain, title_train)
    
    acc_train = calculate_accurancy(Pred_train, ytrain)
    acc_test = calculate_accurancy(Pred_test, ytest) 
    print("acc_train = ", acc_train, ", acc_test = ", acc_test)
    
    sensitivity_test, specificity_test = calculate_sensitivity_specificity(Pred_test, ytest)
    print("Sensitivity (Test):", sensitivity_test)
    print("Specificity (Test):", specificity_test)

    sensitivity_train, specificity_train = calculate_sensitivity_specificity(Pred_train, ytrain)
    print("Sensitivity (Train):", sensitivity_train)
    print("Specificity (Train):", specificity_train)
    
def task1():
    mu0 = [0, 2, 3]
    mu1 = [3, 5, 1]
    mu = [mu0, mu1]
    sigma0 = [2, 1, 2]
    sigma1 = [1, 2, 1]
    sigma = [sigma0, sigma1]
    col = len(mu0)
    n = 1000
    
    x, y, class0, class1 = dataGen.norm_dataset(mu, sigma, n)
    predict(x, y, class0, class1)
    
def task2():
    ellipse_params = {'center_x': 50, 'center_y': 50, 'a': 45, 'b': 45}
    rectangle_params = {'left': 0, 'right': 100, 'bottom': 0, 'top': 70, 'circle_center_x': 50, 'circle_center_y': 50, 'circle_radius': 10}
    N = 1000
    
    x, y, class0, class1 = dataGen.generate_points(N, ellipse_params, rectangle_params)
    predict(x, y, class0, class1)

def classify(classifier, xtrain, xtest, ytrain, ytest):

    print("Train accuracy = ", classifier.score(xtrain, ytrain))
    print("Test accuracy = ", classifier.score(xtest, ytest))
    sensitivity_test, specificity_test = calculate_sensitivity_specificity(classifier.predict(xtest), ytest)
    print("Test sensitivity = ", sensitivity_test)
    print("Test specificity = ", specificity_test)
    
    y_probas_tree = classifier.predict_proba(xtest)
    skplt.metrics.plot_roc(ytest, y_probas_tree, figsize = (10,10))
    plt.show()

    AUC = roc_auc_score(ytest, y_probas_tree[:,1])
    print("AUC:"+str(AUC))

def lab3():
    ellipse_params = {'center_x': 50, 'center_y': 50, 'a': 45, 'b': 45}
    rectangle_params = {'left': 0, 'right': 100, 'bottom': 0, 'top': 70, 'circle_center_x': 50, 'circle_center_y': 50, 'circle_radius': 10}
    N = 1000
    x, y, class0, class1 = dataGen.generate_points(N, ellipse_params, rectangle_params)

    plt.scatter(class0[:, 0], class0[:, 1], marker=".", alpha=0.7, label='1')
    plt.scatter(class1[:, 0], class1[:, 1], marker=".", alpha=0.7, label='2')
    plt.show()

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3)


    print("Tree")
    classifier = DecisionTreeClassifier(max_depth=5)  
    classifier.fit(xtrain, ytrain)
    classify(classifier, xtrain, xtest, ytrain, ytest)

    print("Forest")
    classifier = RandomForestClassifier(max_depth=5)  
    classifier.fit(xtrain, ytrain)
    classify(classifier, xtrain, xtest, ytrain, ytest)

    # FOR loop
    best_auc = 0
    best_num_trees = 0
    num_trees_range = range(1, 301, 10)
    
    for num_trees in num_trees_range:
        forest = RandomForestClassifier(n_estimators=num_trees, random_state=0)
        forest.fit(xtrain, ytrain)
        y_probas = forest.predict_proba(xtest)
        auc = roc_auc_score(ytest, y_probas[:, 1])
        if auc > best_auc:
            best_auc = auc
            best_num_trees = num_trees

    print("Best AUC:", best_auc)
    print("Optimal tree num:", best_num_trees)


if __name__ == "__main__":
    lab3()

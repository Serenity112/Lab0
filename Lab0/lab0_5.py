import sklearn
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train():
    #1
    table_orig = pd.read_csv("csv1.csv")
    df = pd.DataFrame(table_orig)
    X = df.values[:, :-1]
    Y = df.values[:, -1]

    #2
    df['Age'] = preprocessing.scale(df['Age'])

    #3
    df_encoded = pd.get_dummies(df, columns=df.columns[:-1])
    df_encoded.drop(columns=['class']).to_csv('X.csv', index=False)
    df_encoded['class'].to_csv('Y.csv', index=False)

    #4
    numpy_arrayX = pd.DataFrame(pd.read_csv("X.csv")).values
    numpy_arrayY = pd.DataFrame(pd.read_csv("Y.csv")).values
    X = numpy_arrayX
    Y = numpy_arrayY

    #5
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=17)
    model = LogisticRegression(random_state=17).fit(X_train,Y_train)
    Y_pred = model.predict(X_test)
    
    #print(Y_pred)
    print("Точность обычной модели:")
    print(accuracy_score(Y_test, Y_pred))
    
    #6
    param_range = [100, 10, 1, 0.1, 0.01, 0.001]
    logreg = LogisticRegression()
    param_grid = {'C': param_range}
    grid_search = GridSearchCV(logreg, param_grid, cv=5)
    grid_search.fit(X_train, np.ravel(Y_train,order="c"))
    best_C = grid_search.best_params_['C']
    print("Лучшее C:")
    print(best_C)
    best_model = grid_search.best_estimator_
    print("Точность GridSearchCV модели:")
    accuracy = best_model.score(X_test, Y_test)
    print(accuracy)
    
train()


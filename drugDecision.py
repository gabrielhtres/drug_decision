from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load
from graphviz import Graph, Digraph


import numpy as np
import pandas as pd
import graphviz

np.random.seed(25)
arquivo = pd.read_csv('drug200.csv')
rename_columns = {
    "Age" : "Idade",
    "Sex" : "Sexo",
    "BP" : "PS",
    "Cholesterol" : "Colesterol",
    "Na_to_K" : "Sodio_Potassio",
    "Drug" : "Remédio"
}

rename_classes = {
    "drugA" : "A",
    "drugB" : "B",
    "drugC" : "C",
    "drugX" : "X",
    "DrugY" : "Y",
}

rename_sexo = {
    "F" : 0,
    "M" : 1
}

rename_levels = {
    "LOW" : 1,
    "NORMAL" : 2,
    "HIGH" : 3,
}

arquivo = arquivo.rename(columns=rename_columns)
# arquivo

x = arquivo[["Idade", "Sexo", "PS", "Sodio_Potassio", "Colesterol"]]
x["Sexo"] = x["Sexo"].map(rename_sexo)
x["PS"] = x["PS"].map(rename_levels)
x["Colesterol"] = x["Colesterol"].map(rename_levels)
# x.head()

y = arquivo["Remédio"].map(rename_classes)
# y.head()

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.30, stratify=y)

clf = tree.DecisionTreeClassifier(criterion="gini")
#, max_depth=3)'  ''
clf.fit(train_x, train_y)

predict = clf.predict(test_x)
acuracia = accuracy_score(test_y, predict)
print("A acuracia foi de " + str(acuracia * 100) + "%")
# print(classification_report(test_y, predict))

# tree.plot_tree(clf, feature_names=x.columns, class_names=["A", "B", "C", "X", "Y"], filled=True)

# dot_data = tree.export_graphviz(clf, out_file=None, feature_names=x.columns, class_names=["A", "B", "C", "X", "Y"], filled=True, rounded=True, special_characters=True)
# graph = graphviz.Source(dot_data)

# dump(clf, 'modelo_dd.joblib')

# modelo = load('modelo_dd.joblib')

# modelo.fit(train_x, train_y)
# predict = modelo.predict(test_x)
# acuracia = accuracy_score(test_y, predict)
# # print("A acuracia foi de " + str(acuracia * 100) + "%")
# print(classification_report(test_y, predict))

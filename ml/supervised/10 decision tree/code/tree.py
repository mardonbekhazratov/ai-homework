#type: ignore
import os
os.chdir(os.path.dirname(__file__))
import numpy as np
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier

iris = load_iris(as_frame=True)

X = iris.data[['petal length (cm)', 'petal width (cm)']].values
y = iris.target.values

tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X, y)

from sklearn.tree import export_graphviz

export_graphviz(
    tree_clf,
    out_file='iris_tree.dot',
    feature_names=['petal length (cm)', 'petal width (cm)'],
    class_names=iris.target_names,
    rounded=True,
    filled=True
)

from graphviz import Source
Source.from_file('iris_tree.dot')

tree_clf.predict_proba([[5, 1.5]]).round(3)
tree_clf.predict([[5, 1.5]])
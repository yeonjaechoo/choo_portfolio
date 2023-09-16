import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# following youtube..
from time import sleep

#load the iris dataset from sklearn's toy dataset.
iris_dataset = load_iris()

# split the dataframe into a train and test set
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0, shuffle=True)

# create a dataframe using the iris dataset for visualization purpose
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

# create a pair-scatter plot using altair
i_df = iris_dataframe.copy()
i_df['species'] = y_train
alt.Chart(i_df).mark_circle().encode(
    alt.X(alt.repeat('column'), type='quantitative'),
    alt.Y(alt.repeat('row'), type='quantitative'),
    color='species:N'
).properties(
    width=180,
    height=180
).repeat(
    row= ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
    column= ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
)

# now we will build a predictive model using  the K-nearest neighbor algorithm
knn = KNeighborsClassifier(n_neighbors=1)
# we will fit the model using our training set
knn.fit(X_train, y_train)
# using the score method, we will compare the score of model by testing it against the test set data.
knn.score(X_test, y_test)


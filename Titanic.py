# Import Libraries 

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn import tree

# importing dataset 

df = pd.read_csv('/Users/adrikadatta/Downloads/Python/Titanic/DataSet_Titanic.csv')
df.head()

#Separating the predictor attributes: 
X = df.drop('Sobreviviente', axis=1)
X.head()
#Storing the predictor attribute : 
y = df.Sobreviviente
y.head()

# Creating the tree object : 
my_tree = DecisionTreeClassifier(max_depth=2, random_state=42)

# training : 
my_tree.fit(X,y)

# Predicting our sets :
prediction_y = my_tree.predict(X)
# Comparing with real labels : 
print("Accuracy: ", accuracy_score(prediction_y, y))
# We get an accuracy of 80.25% 

# Let's visually check things out : 
cm = confusion_matrix(y, prediction_y)
disp = ConfusionMatrixDisplay(cm, display_labels=my_tree.classes_)
disp.plot()
plt.show()

# Decision Tree: 
plt.figure(figsize=(10,8))
tree.plot_tree(my_tree, filled= True, feature_names=X.columns)
plt.show()

# Vizualising with a Bar Chart : 
relevance = my_tree.feature_importances_
columns = X.columns

sns.barplot(columns, relevance)
plt.title("Relevance of each attribute")
plt.show()







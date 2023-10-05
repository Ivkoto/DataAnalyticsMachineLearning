import warnings as wg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from IPython.display import Image
Image(filename=r'C:\Users\kostovi\source\Python_Mimi\ML\Tutorial_1\iris.png', width=600, height=300)

wg.filterwarnings('ignore')

# load_iris has both the data and the class labels for each sample. Quickly extraction of all of it.
from sklearn.datasets import load_iris
data = load_iris().data        # data is an array where all records are stored

# data variable will be a numpy array of shape (150,4) having 150 samples each having four different attributes. Each class has 50 samples each.
data.shape

# Extract the class labels.
labels = load_iris().target
print(labels)        # First 50 samples belongs to class 0, next 50 samples belong to class 1 and third class as 2

# reshape the labels also to a 2-d array
labels = np.reshape(labels,(150, 1))
#  Concatenate arrays, and use axis=-1 which will concatenate based on the 2nd dimension.
data = np.concatenate([data, labels], axis=-1)
data.shape

# arrange data in a tabular fashion 
# perform some operations and manipulations on the data.
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'species'] # Store the column headings
dataset = pd.DataFrame(data, columns=names)
print(dataset.head(n=5))

# Convert the labels variable's class labels which are numeric values as the flower names or species.
dataset['species'].replace(0, 'Iris-setosa', inplace=True)
dataset['species'].replace(1, 'Iris-versicolor', inplace=True)
dataset['species'].replace(2, 'Iris-virginica', inplace=True)

print(dataset.head(n=55))

# Finding out how all the three flowers look like when visualized and how different they are from each other
# visualizing the data that you loaded above using to find out how much one variable is affected 
# by the other variable (how much correlation is between the two variables)
plt.figure(4, figsize=(10, 8))  # Size of the figure as length and width

# data[:50,0] means first 50 rows for Sepal length (0), and 0 means Sepal length
# data[:50,1] means first 50 rows for Sepal width (1), and 1 means Sepal width
# c='r' is an argument used with the scatter function from the Matplotlib library
# to specify the color of the points in the scatter plot r=red, g=green, b=blue
plt.scatter(data[:50, 0], data[:50, 1], c='r', label='Iris-setosa')           # Taking only first 50 rows for Iris-setosa
plt.scatter(data[50:100, 0], data[50:100, 1], c='g', label='Iris-versicolor') # Taking next 50 rows for Iris-versicolor
plt.scatter(data[100:, 0], data[100:, 1], c='b', label='Iris-virginica')      # Taking next 50 rows for Iris-virginica

plt.xlabel('Sepal length', fontsize = 20)
plt.ylabel('Sepal width', fontsize = 20)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.title('Sepal length vs. Sepal width', fontsize = 20)
plt.legend(prop = {'size': 18})
# prop = {'size': 18} is an argument used with the legend function in Matplotlib
# prop is a dictionary that allows you to specify various properties for the legend.
# {'size': 18} is the dictionary containing a single key-value pair.
# 'size' dict key for font size of the legend, 18 font size to be used for the legend text
plt.show()

# Graph for petal-length and petal-width.
plt.figure(4, figsize=(8, 8))

plt.scatter(data[:50, 2], data[:50, 3], c='r', label='Iris-setosa')
plt.scatter(data[50:100, 2], data[50:100, 3], c='g',label='Iris-versicolor')
plt.scatter(data[100:, 2], data[100:, 3], c='b',label='Iris-virginica')

plt.xlabel('Petal length', fontsize = 15)
plt.ylabel('Petal width', fontsize = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.title('Petal length vs. Petal width', fontsize = 15)
plt.legend(prop = {'size': 20})

plt.show()

## Heatmap
correlation_matrix = dataset.select_dtypes(include=[np.number])
sb.heatmap(correlation_matrix.corr(), annot = True, fmt = '.2f', linewidths = 2)
# Display the heatmap
plt.show()
# annot = True - values of the correlation coefficients should be annotated (displayed) in the heatmap
# fmt='.2f' - format of the annotated values as floating-point numbers with two decimal places (e.g., 0.75)
# Display the number of records per class
print(dataset.groupby('species').size())

# visualize if we need a normalization or splitting to training and testing set our data or not
print(dataset.describe())

# split data to training and testing sets 80% - 20%
from sklearn.model_selection import train_test_split
train_data, test_data, train_label, test_label = train_test_split(
  dataset.iloc[:,:3], dataset.iloc[:,4], test_size=0.2, random_state=42)
  # [:,:3]: This slice selects all rows (:) and the first three columns (:3).
  # In Python indexing, column indices start from 0, so :3 selects columns 0, 1, and 2. 
  # This is a way to select specific columns by specifying a range of column indices.

  # [:,4]: This slice selects all rows (:) and the fifth column (4). In Python indexing, 
  # column indices start from 0, so 4 selects the fifth column. This is a way to select 
  # a single column by specifying its index.

# Print the shape of training and testing data along with its labels.
print(train_data.shape, train_label.shape, test_data.shape, train_label.shape)

Image(filename=r'C:\Users\kostovi\source\Python_Mimi\ML\Tutorial_1\Im1.png', width=400, height=400)

from sklearn.neighbors import KNeighborsClassifier
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings

#Confidence mode - suppress the warnings
warnings.filterwarnings('ignore')

# Dynamically determine the path of the currently executing script
script_directory = os.path.dirname(os.path.abspath(__file__))
# Construct the full path to the CSV
csv_file_path = os.path.join(script_directory, 'Tutorial_2', 'Social_Network_Ads.csv')
# Load Social_Network_Ads.csv file into dataframe
df = pd.read_csv(csv_file_path)

print(df.head(5))

x = df.iloc[:, [2, 3]].values
y = df.iloc[:, -1].values

print ("x DataFrame:")
print(x) 
print ("y DataFrame:")
print(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

# test_size=0.25 => 25% used as the test set, 75% used as training set.
# random_state is used to control the randomness of the data splitting process.
# When it is specified (an integer), it ensures that the data split is reproducible. 
# Setting a fixed random_state value makes code produce the same split every time it will be runned, 
# useful for reproducibility and debugging. In other case the split will be different each time.
x_train, x_test, y_traint, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
print(x.shape, y.shape, x_train.shape, x_test.shape, y_traint.shape, y_test.shape)
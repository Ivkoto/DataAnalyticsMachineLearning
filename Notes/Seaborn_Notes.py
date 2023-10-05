### visualize the diagram preview in VS Code using matplotlib.pyplot .show()

import seaborn as sb
import matplotlib.pyplot as plot
import pandas as pd

dataset = pd.read_excel('C:/Users/kostovi/source/Python_Mimi/Exercise_2_Descriptive_Stats.xlsx')

ax = sb.histplot(dataset, x = 'Month_of_sale')
ax.set(xlabel='Month', ylabel='Sales', title ='Sales per month')

#this will open new window with the bar diagram of the loaded data
plot.show()

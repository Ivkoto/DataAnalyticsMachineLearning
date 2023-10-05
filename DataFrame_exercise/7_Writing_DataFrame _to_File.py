import pandas as pd
import seaborn as sb

diamonds_df_sb = sb.load_dataset('diamonds')

# read result from the url and the loaded data in seaborn
print('\n\nread result from the loaded data in seaborn:\n')
print(diamonds_df_sb.head(n = 5))

# Write the diamonds dataset into a .csv file
diamonds_df_sb.to_csv('diamonds_modified.csv', index = False)

print(diamonds_df_sb.head(n = 5))
import pandas as pd
import seaborn as sb

diamonds_df_sb = sb.load_dataset('diamonds')

# read result from the url and the loaded data in seaborn
print('\n\nread result from the loaded data in seaborn:\n')
print(diamonds_df_sb.head(n = 5))

# function to filter the cut and the color of the diamonds
def is_desired(x):
  return 'yes' if (x['cut'] == 'Ideal' and x['color'] == 'D') else 'no'

# Apply function is_desired and add new column 'desired'
diamonds_df_sb['desired'] = diamonds_df_sb.apply(is_desired, axis = 1)

print('\n\nAdd new column desired by filtering the diamond cut and color:\n')
print(diamonds_df_sb.head(n = 5))
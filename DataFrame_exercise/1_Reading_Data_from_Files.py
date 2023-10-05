import pandas as pd
import seaborn as sb

diamonds_url ='https://raw.githubusercontent.com/TrainingByPackt/Interactive-Data-Visualization-with-Python/master/datasets/diamonds.csv'

diamonds_df = pd.read_csv(diamonds_url)
diamonds_df_sb = sb.load_dataset('diamonds')

# read result from the url and the loaded data in seaborn
print('\n\nread result from the url and the loaded data in seaborn:\n')
print(diamonds_df.head(n = 5))
print(diamonds_df_sb.head(n = 5))

# unique values
print('\n\nunique values:')
print(diamonds_df['cut'].nunique())
print(diamonds_df.color.nunique())
print(diamonds_df.clarity.nunique())

import pandas as pd
import seaborn as sb
import numpy as np

diamonds_url ='https://raw.githubusercontent.com/TrainingByPackt/Interactive-Data-Visualization-with-Python/master/datasets/diamonds.csv'

diamonds_df = pd.read_csv(diamonds_url)
diamonds_df_sb = sb.load_dataset('diamonds')

# read result from the url and the loaded data in seaborn
print('\n\nread result from the url and the loaded data in seaborn:\n')
print(diamonds_df.head(n = 5))
print(diamonds_df_sb.head(n = 5))

# Adding New Columns to a DataFrame
diamonds_df['price_per_carat'] = diamonds_df['price'] / diamonds_df['carat']

print("\n\nAdd New Column 'price_per_carat':\n")
print(diamonds_df.head(n = 5))

# Add column bassed on the price per carat
diamonds_df['price_per_carat_is_high'] = np.where(diamonds_df['price_per_carat'] > 3500, 1, 0)

print("\n\nAdd New Column 'price_per_carat_is_high':\n")
print(diamonds_df.head(n = 5))
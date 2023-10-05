import pandas as pd
import seaborn as sb
import numpy as np
import math

diamonds_url ='https://raw.githubusercontent.com/TrainingByPackt/Interactive-Data-Visualization-with-Python/master/datasets/diamonds.csv'

diamonds_df = pd.read_csv(diamonds_url)
diamonds_df_sb = sb.load_dataset('diamonds')

# read result from the url and the loaded data in seaborn
print('\n\nread result from the url and the loaded data in seaborn:\n')
print(diamonds_df.head(n = 5))
print(diamonds_df_sb.head(n = 5))

diamonds_df['price_per_carat'] = diamonds_df['price'] / diamonds_df['carat']
diamonds_df['price_per_carat_is_high'] = np.where(diamonds_df['price_per_carat'] > 3500, 'true', 'false')

diamonds_df['price']= diamonds_df['price'] * 1.3

# Complex function to round off the price of diamonds
diamonds_df['rounded_price'] = diamonds_df['price'].apply(math.ceil)

print('\n\nAdded new column with rounded prices of diamonds:\n')
print(diamonds_df.head(n = 5))

# Using lambda function to round off the prices
diamonds_df['rounded_price_to_100multiple'] = diamonds_df['price'].apply(lambda x: math.ceil(x/100)*100)

print('\n\nAdd column of rounded prices using lambda function:\n')
print(diamonds_df.head(n = 5))

# User-defined function
def get_100_multiuply_ceil(x):
    return math.ceil(x/100)*100

diamonds_df['rounded_price_to_100multiple'] = diamonds_df['price'].apply(get_100_multiuply_ceil)

print('\n\nAdd column of rounded prices using user defined function:\n')
print(diamonds_df.head(n = 5))
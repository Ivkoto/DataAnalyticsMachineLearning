import pandas as pd
import seaborn as sb
import numpy as np
import math

diamonds_df_sb = sb.load_dataset('diamonds')

# read result from the url and the loaded data in seaborn
print('\n\nread result from the loaded data in seaborn:\n')
print(diamonds_df_sb.head(n = 5))

diamonds_df_sb['price_per_carat'] = diamonds_df_sb['price'] / diamonds_df_sb['carat']

diamonds_df_sb['price_per_carat_is_high'] = np.where(diamonds_df_sb['price_per_carat'] > 3500, 1, 0)

diamonds_df_sb['rounded_price'] = diamonds_df_sb['price'].apply(math.ceil)

def get_100_multiple_ceil(x):
  return math.ceil(x/100)*100

diamonds_df_sb['rounded_price_to_100multiple'] = diamonds_df_sb['price'].apply(get_100_multiple_ceil)

print('\n\n Columns before deleting:\n')
print(diamonds_df_sb.head(n = 5))

# Delete the rounded_price and rounded_price_to_100multiple columns using the drop function
diamonds_df_sb = diamonds_df_sb.drop(columns = ['rounded_price', 'rounded_price_to_100multiple'])

print('\n\n Columns after deleting:\n')
print(diamonds_df_sb.head(n = 5))
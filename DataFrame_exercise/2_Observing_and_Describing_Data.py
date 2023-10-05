import pandas as pd
import seaborn as sb

diamonds_url ='https://raw.githubusercontent.com/TrainingByPackt/Interactive-Data-Visualization-with-Python/master/datasets/diamonds.csv'

diamonds_df = pd.read_csv(diamonds_url)
diamonds_df_sb = sb.load_dataset('diamonds')

#### Convert all the object data type to category types read from the URL
### Identify columns with data type 'object'
#object_columns = diamonds_df.select_dtypes(include=['object']).columns
#diamonds_df[object_columns] = diamonds_df[object_columns].astype('category')

### Convert specific categorical columns back to object data types read from seaborn library
#categorical_columns = ["cut", "color", "clarity"]
#for col in categorical_columns:
#    diamonds_df_sb[col] = diamonds_df_sb[col].astype('object')


# read result from the url and the loaded data in seaborn
print('\n\nread result from the url and the loaded data in seaborn:\n')
print(diamonds_df.head(n = 5))
print(diamonds_df_sb.head(n = 5))

# Count the number of rows and columns in the DataFrame
print('\n\nCount the number of rows and columns in the DataFrame:\n')
print(diamonds_df.shape)

# Summarize the columns to obtain the distribution of variables
# for continuous variables
print('\n\nSummarize the columns for continuous variables:\n')
print(diamonds_df.describe())

# Summarize the columns to obtain the distribution of variables
# for categorical variables
### uncomment print lines below if you skip the converting from object to category or if all the category type ar converted to object dtype
print('\n\nSummarize the columns for categorical variables\n')
print(diamonds_df.describe(include=object))

# To obtain information on the dataset - column types and memory usage
print('\n\nColumn types and memory usage from URL:\n')
print(diamonds_df.info())
print('\n\nColumn types and memory usage from library load:\n')
print(diamonds_df_sb.info())

# Selecting Columns from a DataFrame and Apply Filter
diamonds_low_df = diamonds_df.loc[diamonds_df['cut'] == 'Ideal']

print('\n\nAll rows corresponding to diamonds that have the Ideal cut:\n')
print(diamonds_low_df.head(n = 5))
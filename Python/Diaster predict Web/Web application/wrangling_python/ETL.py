# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

path = r'data'

# load dataset
messages = pd.read_csv(path + '\messages.csv')
categories = pd.read_csv(path + '\categories.csv')

# merge datasets
df = pd.merge(messages,categories,on = 'id')

# create a dataframe of the 36 individual category columns
categories = df['categories'].str.split(pat = ';',expand = True)
category_colnames = [col.split('-')[0] for col in categories.iloc[1].values]
# rename the columns of `categories`
categories.columns = category_colnames

for column in category_colnames:
    # set each value to be the last character of the string
    categories[column] = categories[column].apply(lambda x:x.split('-')[1])

    # convert column from string to numeric
    categories[column] = categories[column].astype(int)

# drop the original categories column from `df`
df = df.drop(columns = 'categories')
# concatenate the original dataframe with the new `categories` dataframe
df = pd.concat([df,categories],axis = 1)

engine = create_engine('sqlite:///' + path + '\ETL_Cleaned.db')
df.to_sql('message',
          engine,
          index=False,
          if_exists='replace')
















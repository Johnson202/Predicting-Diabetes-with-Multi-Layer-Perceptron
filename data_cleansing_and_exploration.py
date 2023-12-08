import warnings
warnings.filterwarnings('ignore')
import pandas as pd



def read_csv():
  #read the diabetes.csv file using pandas library and return it
  df = pd.read_csv("./diabetes.csv")
  return df


def check_null_values():
  df = read_csv()
  #check the null values for each column in the dataset
  return df.isnull().sum()


def count_zeros():
  df = read_csv()
  zero_counts = {}
  df_length = len(df)
  #find out the total number of value 0 present in the each column
  for col in df.columns.values:
      zero_counts[col] = int(df_length - df[col].astype(bool).sum())
  return zero_counts


def correlation():
  df = read_csv()
  #calculate the correlation of the given dataset and round it to 2 decimal places
  return round(df.corr(), 2)
import data_cleansing_and_exploration as t1
from sklearn.model_selection import train_test_split
import numpy as np

#Replace 0 value with NaN
def replace():
  df = t1.read_csv()
  #replace the 0 value with NaN for the columns Glucose, BloodPressure, SkinThickness, Insulin, BMI. and return the updated dataset.
  cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
  dict_to_replace = {}
  for col in cols:
      dict_to_replace[col] = 0
  df.replace(to_replace=dict_to_replace, value=np.nan, inplace=True)
  return df



# Replace Nan values with the mean of the non-missing values by using fillna()
def filling():
  df = replace()
  #replace NaN values with mean of the non-missing values by using fillna() for the columns Glucose, BloodPressure, SkinThickness, Insulin, BMI, and return the updated dataset.
  #set to 2 decimal places
  cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
  for col in cols:
      avg_value = round(df[col].mean(), 2)
      df[col].replace(np.nan, avg_value, inplace=True)
  return df



def split():
  df = filling()
  #define 2 variables for example x and y; assign the independent variables(without Outcome) in to x, and dependent variable(Outcome) in to y.
  cols = df.columns.values
  indep_cols = []
  for col in cols:
      if col != "Outcome":
          indep_cols.append(col)
  x = df[indep_cols]
  y = df['Outcome']
  
  # Split the data into training set(80%) and the testing set(20%), use random_state = 2
  X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=2)
  
  return X_train, X_test, y_train, y_test
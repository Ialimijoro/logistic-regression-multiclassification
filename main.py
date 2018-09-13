import numpy as np 
import pandas as pd
from iris_logistic_regression import *

def run() :
  data = pd.read_csv('data.csv')
  sepal_length_data = np.array(data.sepal_length)
  sepal_width_data = np.array(data.sepal_width)
  petal_length_data = np.array(data.petal_length)
  petal_width_data = np.array(data.petal_width)
  name_data = np.array(data.name)

  irisLogisticRegression = IrisLogisticRegression(sepal_length_data,sepal_width_data,petal_length_data,petal_width_data,name_data)
  irisLogisticRegression.predict(5.1,3.2,1.3,0.2)

if __name__ == '__main__' :
  run()
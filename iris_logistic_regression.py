import numpy as np

class IrisLogisticRegression() : 

  def __init__(self, sepal_length_data, sepal_width_data, petal_length_data ,petal_width_data, name_data):
    self.sepal_length_data = sepal_length_data
    self.sepal_width_data = sepal_width_data
    self.petal_length_data = petal_length_data
    self.petal_width_data = petal_width_data
    self.name_data = name_data
  
  def sigmoid(self, x):
    return (1 / (1 + np.exp(-x)))

  def z(self, thetas, features) :
    return thetas[0] * features[0] + thetas[1] * features[1] + thetas[2] * features[2] + thetas[3] * features[3] + thetas[4] * features[4] 

  def hypothesis(self, thetas, features) :
    return self.sigmoid(self.z(thetas, features))

  def get_name_code(self, name) :
    if name == 'Iris-setosa':
      return 1
    # elif name == 'Iris-versicolor':
    #   return 
    # elif  name == 'Iris-virginica':
    #   return 3
    else :
      return 0

  def cost(self, thetas):
    m = len(self.name_data)
    for i in range(m) :
      features = [ 1, self.sepal_length_data[i], self.sepal_width_data[i], self.petal_length_data[i], self.petal_width_data[i] ]
      cost += self.name_data * np.log(self.hypothesis(thetas, features)) + (1 - self.get_name_code(self.name_data[i])) * np.log(1 - self.hypothesis(thetas, features))
    return - (1/m) * cost
    # return (-1/m)*sum( self.name_data * np.log(self.hypothesis(thetas, features)) + (1 - self.name_data) * np.log(1 - self.hypothesis(thetas, features)) )

  def gradient_descent(self):
    thetas = [0,0,0,0,0]
    thetas_d = [0,0,0,0,0]
    learning_rate = 0.00004
    iterations = 100
    cost = 0
    m = len(self.name_data)

    for i in range(iterations) :
      for i in range(m) :
        features = [ 1, self.sepal_length_data[i], self.sepal_width_data[i], self.petal_length_data[i], self.petal_width_data[i] ]
        cost += self.get_name_code(self.name_data[i]) * np.log(self.hypothesis(thetas, features)) + (1 - self.get_name_code(self.name_data[i])) * np.log(1 - self.hypothesis(thetas, features))

        thetas_d[0] += ( self.hypothesis(thetas, features) -  self.get_name_code(self.name_data[i]) )
        thetas_d[1] += ( self.hypothesis(thetas, features) -  self.get_name_code(self.name_data[i]) ) * self.sepal_length_data[i]
        thetas_d[2] += ( self.hypothesis(thetas, features) -  self.get_name_code(self.name_data[i]) ) * self.sepal_width_data[i]
        thetas_d[3] += ( self.hypothesis(thetas, features) -  self.get_name_code(self.name_data[i]) ) * self.petal_length_data[i]
        thetas_d[4] += ( self.hypothesis(thetas, features) -  self.get_name_code(self.name_data[i]) ) * self.petal_width_data[i]
      
      cost = - (1/m) * cost
      # print("theta_0 : {}, theta_1 : {}, theta_2 {}, theta_3 : {}, theta_4 : {}, cost : {}".format(thetas[0], thetas[1], thetas[2], thetas[3], thetas[4], cost))
      thetas[0] = thetas[0] - learning_rate * thetas_d[0]
      thetas[1] = thetas[1] - learning_rate * thetas_d[1]
      thetas[2] = thetas[2] - learning_rate * thetas_d[2]
      thetas[3] = thetas[3] - learning_rate * thetas_d[3]
      thetas[4] = thetas[4] - learning_rate * thetas_d[4]
    
    return thetas
  
  def predict(self, sepal_length, sepal_width, petal_length, petal_width) :
    thetas = self.gradient_descent()
    p = self.hypothesis(thetas, [1, sepal_length, sepal_width, petal_length, petal_width])

    print("Given the following\nsepal length: {}".format(sepal_length))
    print("sepal width: {}".format(sepal_width))
    print("petal length: {}".format(petal_length))
    print("petal width: {}".format(petal_width))

    print("probability of being an Iris-setosa: {}".format(p))
    
    if p >= 0.5 : 
      print('We can say that the flower is an Iris Cetosa')
    else : 
      print('We can say that the flower is not an Iris Cetosa')
    # print(self.hypothesis(thetas, [1, sepal_length, sepal_width, petal_length, petal_width]))
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from sklearn import svm
image = misc.imread('datasets/bezier_diagonal.png')

def training_data(universe, n = 10):
  xs = np.random.randint(universe.shape[0], size=n)
  ys = np.random.randint(universe.shape[1], size=n)
  results = np.array([bool(universe[xs[i]][ys[i]]) for i in range(n)])
  return np.column_stack((xs, ys, results))

def showFitGraph(original, prediction, training_data=None):
  width, height = original.shape

  f, ax = plt.subplots(1, 2)
  left, right = ax 
  left.set_title('Training data')
  right.set_title('Prediction')


  left.imshow(original, cmap='Greys')
  #data = data[:100]
  if not None == training_data:
    left.plot(
      [row[0] for row in training_data], 
      [row[1] for row in training_data], 
      'go'
    )
  left.axis([0, width-1, 0, height-1])


  right.imshow(prediction, cmap='Greys')
  right.axis([0, width-1, 0, height-1])
  right.axis([0, width-1, 0, height-1])

  plt.show()

def learn(data):
  inputs  = [ x[:2] for x in data ]
  targets = [ x[2]  for x in data ]
  clf = svm.SVC(gamma=0.001, C=100.)
  clf.fit(inputs, targets)
  return clf.predict

def main():
  data = training_data(image, 1000)
  predict = learn(data)

  xs = np.arange(0, image.shape[0], 1)
  ys = np.arange(0, image.shape[1], 1)


  prediction = predict([[x, y] for x in xs for y in ys]).reshape(image.shape)
  print(prediction)
  showFitGraph(image, prediction, data)

if __name__ == '__main__':
  main()

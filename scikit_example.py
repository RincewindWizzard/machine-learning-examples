import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import misc
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
 


def training_data(universe, n = 10):
  width, height = universe.shape
  if n > width*height: raise ValueError("Not enough input space for sample size!")

  data = [[x, y, universe[x][y]] for x in range(width) for y in range(height)]
  data = random.sample(data, n)

  return data

def normalize_data(data, width, height):
  return [ [d[0]/width, d[1]/height] + d[2:] for d in data]

def difference_image(imga, imgb):
  if not imga.shape == imgb.shape: raise ValueError("The two images have unequal shapes!")

  errors = 0
  diff = []
  for x in range(imga.shape[0]):
    row = []
    diff.append(row)
    for y in range(imga.shape[1]):
      correct = bool(imga[x][y]) == bool(imgb[x][y])
      if not correct: errors += 1
      row.append(int(not correct))
  return np.array(diff), errors

def showFitGraph(original, *predictions, training_data=None):
  width, height = original.shape

  count = 1 + len(predictions)
  rowsize = 6
  cols = math.ceil(count / rowsize)

  f, ax = plt.subplots(cols, min(count, rowsize))
  left = ax[(0, 0) if cols > 1 else 0]
  left.set_title('Training data')
  


  left.imshow(original, cmap='Greys')
  #data = data[:100]
  if not None == training_data:
    left.set_title(
      'Training data ({:0.2f} % of area covered)'.format(
        100 * len(training_data) / (width*height) ))
    left.plot(
      [row[0] for row in training_data], 
      [row[1] for row in training_data], 
      'g,'
    )
  left.axis([0, width-1, 0, height-1])

  for i, (title, prediction) in enumerate(predictions):
    graph = ax[((i + 1) // rowsize, (i + 1) % rowsize)  if cols > 1 else (i+1)]
    graph.imshow(prediction, cmap='Greys')
    graph.axis([0, width-1, 0, height-1])


    cmap = plt.cm.Wistia
    # Get the colormap colors
    my_cmap = cmap(np.arange(cmap.N))

    # Set alpha
    my_cmap[:,-1] = np.linspace(0, 0.5, cmap.N)

    # Create new colormap
    my_cmap = ListedColormap(my_cmap)
    
    diffimg, errors = difference_image(original, prediction)
    graph.set_title('{} ({:0.2f} % accurancy)'.format(title, 100 - 100* errors / (width*height)))
    #right.imshow(diffimg, cmap=my_cmap)

  plt.show()



def learnSVM(data):
  inputs  = [ x[:2] for x in data ]
  targets = [ x[2]  for x in data ]
  clf = svm.SVC(gamma=0.001, C=100.)
  clf.fit(inputs, targets)
  return clf.predict

def learnMLP(hidden_layer_sizes=(4)):
  def learn(data):
    inputs  = [ x[:2] for x in data ]
    targets = [ int(bool(x[2]))  for x in data ]

    scaler = StandardScaler()  
    # Don't cheat - fit only on training data
    scaler.fit(inputs)  
    data = scaler.transform(inputs)  


    clf = MLPClassifier(
      solver='lbfgs', 
      alpha=1e-5, 
      hidden_layer_sizes=hidden_layer_sizes, 
      random_state=1
    )
    clf.fit(inputs, targets)

    def predict(test_data):
      # apply same transformation to test data
      test_data = scaler.transform(test_data)
      return clf.predict(test_data)
    return predict
  return learn

def test(image, learn, data, normalize=False):
  if normalize:
    data = normalize_data(data, *image.shape)

  predict = learn(data)

  if normalize:
    xs = np.arange(0, 1, 1/image.shape[0])
    ys = np.arange(0, 1, 1/image.shape[1])
  else:
    xs = np.arange(0, image.shape[0], 1)
    ys = np.arange(0, image.shape[1], 1)

  prediction = predict([[x, y] for x in xs for y in ys]).reshape(image.shape)
  return prediction

def main():
  image = misc.imread('datasets/bezier_diagonal_2.png')
  data = training_data(image, 1000)

  images = [('SVM', test(image, learnSVM, data))]
  for i in range(1):
    images.append((
      'MLP', 
      test(
        image, 
        learnMLP(
          hidden_layer_sizes = (10*i+2, 10*i+2, 10*i+2, 10*i+2,)
      ), data, normalize=True)))

  showFitGraph(image, *images, training_data=data)
if __name__ == '__main__':
  main()

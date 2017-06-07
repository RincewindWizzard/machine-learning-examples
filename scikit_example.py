import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import misc
from sklearn import svm


def training_data(universe, n = 10):
  width, height = universe.shape
  if n > width*height: raise ValueError("Not enough input space for sample size!")

  data = [[x,y,universe[x][y]] for x in range(width) for y in range(height)]
  data = random.sample(data, n)

  return data

def difference_image(imga, imgb):
  if not imga.shape == imgb.shape: raise ValueError("Th two images have unequal shapes!")

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

def showFitGraph(original, prediction, training_data=None):
  width, height = original.shape

  f, ax = plt.subplots(1, 2)
  left, right = ax
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


  right.imshow(prediction, cmap='Greys')
  right.axis([0, width-1, 0, height-1])


  cmap = plt.cm.Wistia
  # Get the colormap colors
  my_cmap = cmap(np.arange(cmap.N))

  # Set alpha
  my_cmap[:,-1] = np.linspace(0, 0.5, cmap.N)

  # Create new colormap
  my_cmap = ListedColormap(my_cmap)
  
  diffimg, errors = difference_image(original, prediction)
  right.set_title('Prediction ({:0.2f} % accurancy)'.format(100 - 100* errors / (width*height)))
  right.imshow(diffimg, cmap=my_cmap)

  plt.show()

def learn(data):
  inputs  = [ x[:2] for x in data ]
  targets = [ x[2]  for x in data ]
  clf = svm.SVC(gamma=0.001, C=100.)
  clf.fit(inputs, targets)
  return clf.predict


def test(image, learn, sample_size=1000):
  data = training_data(image, sample_size)
  predict = learn(data)

  xs = np.arange(0, image.shape[0], 1)
  ys = np.arange(0, image.shape[1], 1)


  prediction = predict([[x, y] for x in xs for y in ys]).reshape(image.shape)
  showFitGraph(image, prediction, data)

def main():
  image = misc.imread('datasets/bezier_diagonal_2.png')
  test(image, learn, sample_size=500)

if __name__ == '__main__':
  main()

import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import misc
import argparse
import importlib
from datetime import datetime


def training_data(universe, n = 10):
  """
  * This function draws a training dataset from the input universe,
  * which is a numpy.array with a known shape.
  * The second argument specifies how big the trainingset should be.
  """
  width, height = universe.shape
  if n > width*height: raise ValueError("Not enough input space for sample size!")

  data = [[x, y, universe[x][y]] for x in range(width) for y in range(height)]
  data = random.sample(data, n)

  return data

def normalize_data(data, max_x, max_y):
  """ Normalize your data points to the interval [0:1] """
  return [ [d[0]/max_x, d[1]/max_y] + d[2:] for d in data]


def difference_image(imga, imgb):
  """
  * Compares two images and returns the pixels, that are different
  """
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


def showResults(original, *predictions, training_data=None):
  """
  * Draw the original universe (an image array) side by side with the prediction images
  * from the learning algorithms.
  * prediction is a list of tuples (title, image).
  * If you specify the training data, green pixels are drawn in the original image 
  * showing the positions of the training data points.
  """
  width, height = original.shape

  # show graphs in multiple rows if not enough space available
  count = 1 + len(predictions)
  rowsize = 6
  cols = math.ceil(count / rowsize)

  f, ax = plt.subplots(cols, min(count, rowsize))
  left = ax[(0, 0) if cols > 1 else 0]
  left.set_title('Training data')

  left.imshow(original, cmap='binary')

  # if training data given, show them
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

  # draw a graph for every prediction
  for i, (title, prediction) in enumerate(predictions):
    graph = ax[((i + 1) // rowsize, (i + 1) % rowsize)  if cols > 1 else (i+1)]
    graph.imshow(prediction, cmap='binary')
    graph.axis([0, width-1, 0, height-1])


    # create a colormap from alpha to black
    cmap = plt.cm.Wistia
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:,-1] = np.linspace(0, 0.5, cmap.N)
    my_cmap = ListedColormap(my_cmap)
    
    diffimg, errors = difference_image(original, prediction)
    graph.set_title(
      '{} ({:0.2f} % accurancy)'.format(
        title, 
        100 - 100* errors / (width*height)
    ))

  result_path = os.path.join(
    'results',
    datetime.now().isoformat() + '.png'
  )
  plt.savefig(result_path)
  plt.show()


def test(image, learn, data, normalize=False):
  """
  * Generate a prediction image for a learning algorithm using given training data set.
  """
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

def main(img_path, training_size, *algorithms):
  # read an image
  image = np.flipud(misc.imread(img_path, flatten=True))

  # choose a training set
  data = training_data(image, training_size)

  from svm_solver import learn as learnSVM
  from mlp_solver import learn as learnMLP

  images = []
  
  for algorithm in algorithms:
    module = importlib.import_module(algorithm)   
    normalize = module.normalize if hasattr(module, 'normalize') else False
    image_result = test(image, module.learn, data, normalize=normalize)
    images.append((module.__name__, image_result))

  
  showResults(image, *images, training_data=data)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Machine learning tester')
  parser.add_argument('image_path', type=str,
                      help='Path to an image of the known universe.')
  parser.add_argument('training_size', metavar='n', type=int,
                      help='How many elements should be in the training data set?')

  parser.add_argument('algorithm', type=str, nargs='+',
                      help='Specify a module, which contains the learning algorithm.')


  args = parser.parse_args()
  main(args.image_path, args.training_size, *args.algorithm)

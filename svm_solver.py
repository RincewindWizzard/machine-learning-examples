from sklearn import svm


def learn(data, gamma=0.001, C=100.):
  inputs  = [ x[:2] for x in data ]
  targets = [ x[2]  for x in data ]
  clf = svm.SVC(gamma=gamma, C=C)
  clf.fit(inputs, targets)
  return clf.predict

from sklearn.neural_network import MLPClassifier
# Tell the test suite, that we want normaluzed data
normalize = True

def learn(data, hidden_layer_sizes=(400, 40, 40)):
  inputs  = [ x[:2] for x in data ]
  targets = [ int(bool(x[2]))  for x in data ]


  clf = MLPClassifier(
    solver='lbfgs', 
    alpha=1e-5, 
    hidden_layer_sizes=hidden_layer_sizes, 
    random_state=1
  )
  clf.fit(inputs, targets)

  return clf.predict

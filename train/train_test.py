from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.utils import shuffle
from features.circular_histogram_depth_difference import CircularHistogramDepthDifference
from features.ri_hog import RIHOG
import numpy as np
from os import listdir
from os.path import isfile, join
from multiprocessing import Process, Value, Array, Pool

#c = CircularHistogramDepthDifference(block_size=8, num_rings=3, ring_dist=10, num_blocks_first_ring=6)
c = RIHOG(num_spatial_bins=4, delta_radius=6, num_orientation_bins=13, normalize=True, normalize_threshold=0.2, gaussian_filter=False, sigma=1, var_feature=True, var_split=32)

def getPositiveSample(f):
  name = 'data/pos/' + f
  try:
    data = c.process_image(name)
    return data, 1
  except:
    print('Failed file: ' + name)
    return [], -1

def getNegativeSample(f):
  name = 'data/neg/' + f
  try:
    data = c.process_image(name)
    return data, 0
  except:
    print('Failed file: ' + name)
    return [], -1

if __name__ == "__main__":

  clf = AdaBoostClassifier(n_estimators=500)

  def listFiles(mypath):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.lower().endswith('.png')]
    return onlyfiles

  def getSamples():
    p = Pool(4)
    pos = np.asarray(p.map(getPositiveSample, listFiles('data/pos/')))
    Yp = np.asarray(pos)[:,1]
    Ip = Yp >= 0
    Yp = Yp[Ip]
    Xp = np.asarray(pos)[:,0]
    Xp = Xp[Ip]

    neg = np.asarray(p.map(getNegativeSample, listFiles('data/neg/')))
    Yn = np.asarray(neg)[:,1]
    In = Yn >= 0
    Yn = Yn[In]
    Xn = np.asarray(neg)[:,0]
    Xn = Xn[In]

    X = np.concatenate((Xp, Xn)).tolist()
    y = np.concatenate((Yp, Yn)).tolist()

    return X, y

  X, y = getSamples()

  X, y = shuffle(X, y, random_state=1)

  from sklearn.tree import _tree

  def tree_to_code(tree, feature_names):
      tree_ = tree.tree_
      feature_name = [
          feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
          for i in tree_.feature
      ]
      print("def tree({}):".format(", ".join(feature_names)))

      def recurse(node, depth):
          indent = "  " * depth
          if tree_.feature[node] != _tree.TREE_UNDEFINED:
              name = feature_name[node]
              threshold = tree_.threshold[node]
              print("{}if {} <= {}:".format(indent, name, threshold))
              recurse(tree_.children_left[node], depth + 1)
              print("{}else:  # if {} > {}".format(indent, name, threshold))
              recurse(tree_.children_right[node], depth + 1)
          else:
              print("{}return {}".format(indent, tree_.value[node]))

      recurse(0, 1)


  clf.fit(X, y)

  tree_to_code(clf.estimators_[0], [str(x) for x in range(0, 60)])
  score = clf.score(X, y)
  print(str(score))

  estimator = clf.estimators_[0]
  n_nodes = estimator.tree_.node_count
  children_left = estimator.tree_.children_left
  children_right = estimator.tree_.children_right
  feature = estimator.tree_.feature
  threshold = estimator.tree_.threshold


  # The tree structure can be traversed to compute various properties such
  # as the depth of each node and whether or not it is a leaf.
  node_depth = np.zeros(shape=n_nodes)
  is_leaves = np.zeros(shape=n_nodes, dtype=bool)
  stack = [(0, -1)]  # seed is the root node id and its parent depth
  while len(stack) > 0:
      node_id, parent_depth = stack.pop()
      node_depth[node_id] = parent_depth + 1

      # If we have a test node
      if (children_left[node_id] != children_right[node_id]):
          stack.append((children_left[node_id], parent_depth + 1))
          stack.append((children_right[node_id], parent_depth + 1))
      else:
          is_leaves[node_id] = True

  print("The binary tree structure has %s nodes and has "
        "the following tree structure:"
        % n_nodes)
  for i in range(n_nodes):
      if is_leaves[i]:
          print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
      else:
          print("%snode=%s test node: go to node %s if X[:, %s] <= %ss else to "
                "node %s."
                % (node_depth[i] * "\t",
                  i,
                  children_left[i],
                  feature[i],
                  threshold[i],
                  children_right[i],
                  ))


  scores = cross_val_score(clf, X, y, cv=5)

  print(str(scores))
  print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

  predicted = cross_val_predict(clf, X, y, cv=5)
  pscore = metrics.accuracy_score(y, predicted) 
  print("Predict Accuracy: %0.2f" % pscore)

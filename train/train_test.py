from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.utils import shuffle
from features.circular_histogram_depth_difference import CircularHistogramDepthDifference

clf = AdaBoostClassifier(n_estimators=5)

c = CircularHistogramDepthDifference(block_size=8, num_rings=3, ring_dist=10, num_blocks_first_ring=6)

def getSamples():
  X = []
  y = []
  i = 1
  while True:
    name = 'data/pos/' + str(i) + '.PNG'
    i += 1
    try:
      data = c.process_image(name)
      X.append(data)
      y.append(1)
    except FileNotFoundError:
      break
    except:
      print('Failed file: ' + name)
      continue
  i = 1
  while True:
    name = 'data/neg/' + str(i) + '.PNG'
    i += 1
    try:
      data = c.process_image(name)
      X.append(data)
      y.append(0)
    except FileNotFoundError:
      break
    except:
      print('Failed file: ' + name)
      continue
  return X, y

X, y = getSamples()

X, y = shuffle(X, y, random_state=1)

#clf.fit(X, y)

#score = clf.score(X, y)
#print(str(score))

scores = cross_val_score(clf, X, y, cv=5)

print(str(scores))
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

predicted = cross_val_predict(clf, X, y, cv=5)
pscore = metrics.accuracy_score(y, predicted) 
print("Predict Accuracy: %0.2f" % pscore)

import cv2
import numpy as np
from features.circular_histogram_depth_difference import CircularHistogramDepthDifference
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

clf = cv2.ml.Boost_create()
clf.setWeakCount(5)
clf.setBoostType(cv2.ml.BOOST_REAL)

c = CircularHistogramDepthDifference(block_size=8, num_rings=3, ring_dist=10, num_blocks_first_ring=6, dist_mult=1.1)

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
X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.int32)

X, y = shuffle(X, y, random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

clf.train(X_train, cv2.ml.ROW_SAMPLE, y_train)

acc = np.sum((y_train == clf.predict(X_train)[1].flatten()).astype(np.int))/len(y_train)
clf.save('test.xml')
print("Training Accuracy: %0.5f" % acc)
acc = np.sum((y_test == clf.predict(X_test)[1].flatten()).astype(np.int))/len(y_test)
print("Test Accuracy: %0.5f" % acc)

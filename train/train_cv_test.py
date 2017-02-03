import cv2
import numpy as np
from features.ri_hog import RIHOG
from features.circular_histogram_depth_difference import CircularHistogramDepthDifference
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from os import listdir
from os.path import isfile, join
from multiprocessing import Process, Value, Array, Pool

clf = cv2.ml.Boost_create()
clf.setWeakCount(500)
clf.setBoostType(cv2.ml.BOOST_REAL)

#c = CircularHistogramDepthDifference(block_size=8, num_rings=3, ring_dist=10, num_blocks_first_ring=6, dist_mult=1.1)
c = RIHOG(num_spatial_bins=4, delta_radius=6, num_orientation_bins=13, normalize=True, normalize_threshold=0.2, gaussian_filter=False, sigma=2, var_feature=True, var_split=32)

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

def trainClf():
  def listFiles(mypath):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.lower().endswith('.png')]
    return onlyfiles

  def getSamples():
    p = Pool(4)
    pos = np.asarray(p.map(getPositiveSample, listFiles('data/pos/')))
    #pos = []
    #for f in listFiles('data/pos/'):
    #  pos.append(getPositiveSample(f))
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
  X = np.asarray(X, dtype=np.float32)
  y = np.asarray(y, dtype=np.int32)

  X, y = shuffle(X, y, random_state=1)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=1)

  clf.train(X_train, cv2.ml.ROW_SAMPLE, y_train)

  acc = np.sum((y_train == clf.predict(X_train)[1].flatten()).astype(np.int))/len(y_train)
  clf.save('test.xml')
  print("Training Accuracy: %0.5f" % acc)
  acc = np.sum((y_test == clf.predict(X_test)[1].flatten()).astype(np.int))/len(y_test)
  print("Test Accuracy: %0.5f" % acc)
  return clf

if __name__ == "__main__":
  trainClf()

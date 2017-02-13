import cv2
import numpy as np
from features.ri_hog import RIHOG
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from multiprocessing import Process, Value, Array, Pool
import time
from train_cv_test import trainClf, getDescriptor

blockSize = 52
halfBlockSize = blockSize/2

data = misc.imread('data/test/test6.png', flatten=True, mode='RGB')
data = np.pad(data, ((blockSize, blockSize), (blockSize, blockSize)), 'constant', constant_values=255)

c = getDescriptor()

def scan(tup):
  x = tup[0]
  y = tup[1]
  roi = data[int(y-halfBlockSize):int(y+halfBlockSize), int(x-halfBlockSize):int(x+halfBlockSize)]
  xx = c.process_data(roi, draw_regions=False)
  return xx

if __name__ == "__main__":

  plt.imshow(data)
  plt.show()

  clf = trainClf()

  print('Starting test')

  detects = []
  fig,ax = plt.subplots(1)

  ax.imshow(data)
  
  starttime = time.time()

  p = Pool(3)
  xylst = []
  for y in range(138+blockSize, data.shape[0] - blockSize - blockSize, 2):
    for x in range(121+blockSize, data.shape[1] - blockSize - blockSize, 2):
      xylst.append((x, y))
      # roi = data[int(y-halfBlockSize):int(y+halfBlockSize), int(x-halfBlockSize):int(x+halfBlockSize)]
      # xx, drawing = c.process_data(roi, draw_regions=True)
      # person = clf.predict(xx.astype('float32'))[0] > 0.5
      # if person:
      #   detects.append((x, y))
      #   print("Person at x: %d, y: %d" % (x - blockSize, y - blockSize))

  results = np.asarray(p.map(scan, xylst))
  predict = []
  for n in range(0, len(results)):
    xx = results[n]
    predict.append(clf.predict(np.asarray(xx).astype('float32'))[0])
  results = np.asarray(predict)
  valid = results[:] >= 0.5
  detects = np.asarray(xylst)[valid]

  elapsedtime = time.time() - starttime
  print('searched image in ' + str(elapsedtime) + ' seconds')

  for x, y in detects:
    rect = patches.Rectangle((x-1,y-1), 2, 2, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
  plt.show()

  print("Done")


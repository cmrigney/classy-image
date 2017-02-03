import cv2
import numpy as np
from features.ri_hog import RIHOG
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.patches as patches

if __name__ == "__main__":

  blockSize = 52
  halfBlockSize = blockSize/2

  data = misc.imread('data/test/test7.png', flatten=True, mode='RGB')
  data = np.pad(data, ((blockSize, blockSize), (blockSize, blockSize)), 'constant', constant_values=255)

  plt.imshow(data)
  plt.show()

  from train_cv_test import trainClf

  clf = trainClf()
  c = RIHOG(num_spatial_bins=4, delta_radius=6, num_orientation_bins=13, normalize=True, normalize_threshold=0.2, gaussian_filter=False, sigma=2, var_feature=True, var_split=32)

  print('Starting test')

  detects = []
  fig,ax = plt.subplots(1)

  ax.imshow(data)

  for y in range(138+blockSize, data.shape[0] - blockSize - blockSize, 2):
    for x in range(121+blockSize, data.shape[1] - blockSize - blockSize, 2):
      roi = data[int(y-halfBlockSize):int(y+halfBlockSize), int(x-halfBlockSize):int(x+halfBlockSize)]
      xx, drawing = c.process_data(roi, draw_regions=True)
      person = clf.predict(xx.astype('float32'))[0] > 0.5
      if person:
        detects.append((x, y))
        print("Person at x: %d, y: %d" % (x - blockSize, y - blockSize))

  for x, y in detects:
    rect = patches.Rectangle((x-1,y-1), 2, 2, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
  plt.show()

  print("Done")


import cv2
import numpy as np
from features.ri_hog import RIHOG
from scipy import misc
import matplotlib.pyplot as plt

blockSize = 52
halfBlockSize = blockSize/2

data = misc.imread('data/test/test6.png', flatten=True)
data = np.pad(data, ((blockSize, blockSize), (blockSize, blockSize)), 'constant', constant_values=255)

plt.imshow(data)
plt.show()

from train_cv_test import getClf

clf = getClf()
c = RIHOG(num_spatial_bins=4, delta_radius=6, num_orientation_bins=7, normalize=True, normalize_threshold=0.2)

print('Starting test')

for y in range(138+blockSize, data.shape[0] - blockSize - blockSize, 2):
  for x in range(121+blockSize, data.shape[1] - blockSize - blockSize, 2):
    roi = data[int(y-halfBlockSize):int(y+halfBlockSize), int(x-halfBlockSize):int(x+halfBlockSize)]
    xx, drawing = c.process_data(roi, draw_regions=True) 
    person = clf.predict(xx.astype('float32'))[0] > 0.5
    if person:
      print("Person at x: %d, y: %d" % (x - blockSize, y - blockSize))

print("Done")


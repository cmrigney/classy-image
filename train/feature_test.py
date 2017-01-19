from features.circular_histogram_depth_difference import CircularHistogramDepthDifference
from features.ri_hog import RIHOG
import matplotlib.pyplot as plt
import cv2

c = CircularHistogramDepthDifference(block_size=8, num_rings=3, ring_dist=10, num_blocks_first_ring=6)

features, drawing = c.process_image('data/test/test.png', True)

#plt.imshow(drawing)
#plt.show()

import numpy as np
def graph(formula, x_range):
  x = np.array(x_range)
  y = eval(formula)
  plt.plot(x,y)

#plt.autoscale(False)
#graph('.5*x', range(0, 11))
#graph('2*x', range(0, 11))
#graph('x', range(0, 11))
#plt.show()

rhog = RIHOG(num_spatial_bins=5, delta_radius=10, num_orientation_bins=9)

img = cv2.imread('data/test/test3.png', 0)
sx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=1)
sy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=1)
s = np.abs(sx) + np.abs(sy)
plt.imshow(s)
plt.show()

features, drawing = rhog.process_image('data/test/test3.png', True)

plt.imshow(drawing)
plt.show()

plt.plot(features)
plt.show()

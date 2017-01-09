import numpy as np
import math
from .base_feature import BaseFeature

class CircularHistogramDepthDifference(BaseFeature):
  def __init__(self, block_size=8, num_rings=3, num_blocks_first_ring=8, block_increaser=4, ring_dist=10, dist_mult=1.5):
    BaseFeature.__init__(self)
    self.block_size = block_size
    self.num_rings = num_rings
    self.num_blocks_first_ring = num_blocks_first_ring
    self.block_increaser = block_increaser
    self.ring_dist = ring_dist
    self.dist_mult = dist_mult
  
  def process_data(self, data, draw_regions):
    drawing = data.copy()
    features = np.asarray([])
    width = data.shape[1]
    height = data.shape[0]
    cx1 = width/2 - self.block_size/2
    cy1 = height/2 - self.block_size/2
    cx = width/2
    cy = height/2
    d_center = np.mean(data[int(cx1):int(cx1+self.block_size), int(cy1):int(cy1+self.block_size)])
    if draw_regions:
      drawing[int(cx1):int(cx1+self.block_size), int(cy1):int(cy1+self.block_size)] = drawing[int(cx1):int(cx1+self.block_size), int(cy1):int(cy1+self.block_size)]/2 
    for ring in range(0, self.num_rings):
      ring_features = []
      dist = (ring+1) * self.ring_dist
      num_blocks = self.num_blocks_first_ring + self.block_increaser * ring
      alpha = 2.*math.pi / float(num_blocks)
      for block in range(0, num_blocks):
        theta = alpha * block
        x = dist * math.sin(theta) + cx - self.block_size/2.
        y = dist * math.cos(theta) + cy - self.block_size/2.
        d = np.mean(data[int(x):int(x+self.block_size), int(y):int(y+self.block_size)])
        if draw_regions:
          drawing[int(x):int(x+self.block_size), int(y):int(y+self.block_size)] = drawing[int(x):int(x+self.block_size), int(y):int(y+self.block_size)]/2
        if math.isnan(d):
          raise Exception('Uh oh')
        ring_features.append(max(d_center - d, 0))
      hist, edges = np.histogram(np.asarray(ring_features), bins=[0, 5, 8, 12, 16])
      features = np.concatenate((features, hist))
    if draw_regions:
      return features, drawing
    else:
      return features


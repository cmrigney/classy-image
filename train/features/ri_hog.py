import numpy as np
import math
from .base_feature import BaseFeature
import cv2

def _norm_and_mult(cx, cy, x, y, magnitude, result):
  result[cy + y, cx + x] = result[cy + y, cx + x] / magnitude
  result[cy + y, cx + x] *= 255

def _norm_and_clip(cx, cy, x, y, magnitude, result, data):
  result[cy + y, cx + x] = data[cy + y, cx + x] / magnitude
  if result[cy + y, cx + x] > .2:
    result[cy + y, cx + x] = .2
  return result[cy + y, cx + x]**2

class RIHOG(BaseFeature):
  def __init__(self, num_spatial_bins=4, delta_radius=4, num_orientation_bins=13, normalize=True, normalize_threshold=0.2):
    BaseFeature.__init__(self)
    self.delta_radius = delta_radius
    self.num_orientation_bins = num_orientation_bins
    self.num_spatial_bins = num_spatial_bins
    self.normalize = normalize
    self.normalize_threshold = normalize_threshold
    self.lut = self._build_lut(64)

  def _build_lut(self, resolution):
    lut = []
    for r in range(0, resolution):
      y = r / (resolution - 1)
      x = math.sqrt(1 - y*y)
      lut.append(x)
    return lut

  def _get_circle_point(self, y, radius):
    if radius == 0:
      return (0, 0)
    resolution = len(self.lut)
    nY = (y / radius)
    rY = nY  * (resolution - 1)
    if rY >= len(self.lut):
      return (0, 0)
    nX = self.lut[int(rY)]
    return (int(nX * radius), y)
  
  def _handle_pixel(self, cx, cy, x, y, data, drawing, draw_regions):
    if draw_regions:
      drawing[cy + y, cx + x] /= 2
    lowerY = .5*abs(x)
    upperY = 2*abs(x)
    absY = abs(y)
    if absY >= lowerY and absY <= upperY:
      dir = [1, 1]
    elif absY < lowerY:
      dir = [1, 0]
    else:
      dir = [0, 1]

    if x < 0:
      dir[0] *= -1
    if y < 0:
      dir[1] *= -1
    
    px = cx + x
    py = cy + y

    gyl = (-dir[1], dir[0])
    gyr = (dir[1], -dir[0])

    gxl = (dir[0], dir[1])
    gxr = (-dir[0], -dir[1])

    gx = data[py + gxr[1], px + gxr[0]] * -1 + data[py + gxl[1], px + gxl[0]]
    gy = data[py + gyl[1], px + gyl[0]] * -1 + data[py + gyr[1], px + gyr[0]]

    if gx == 0:
      gx = 1
    if gy == 0:
      gy = 1
    
    g = math.sqrt(gx**2 + gy**2)
    theta = math.atan(gy/gx) * 180./math.pi

    if draw_regions:
      drawing[py, px] = g

    return (g, theta)
  
  def _bin_values(self, bins, tup):
    g = tup[0]
    theta = tup[1]
    idx = (theta % 180)/180
    idx *= len(bins)
    idx -= 0.5
    lowerIdx = math.floor(idx) % len(bins)
    higherIdx = math.ceil(idx) % len(bins)
    if lowerIdx == higherIdx:
      bins[lowerIdx] += g
      return
    
    lowerPercent = 1 - (abs(idx - lowerIdx) % 1)
    higherPercent = 1 - (abs(higherIdx - idx) % 1)

    bins[lowerIdx] += g*lowerPercent
    bins[higherIdx] += g*higherPercent
  
  def _normalize_features(self, features):
    normFeatures = np.asarray([])
    for spatial_bin in range(0, self.num_spatial_bins):
      startFeatureIdx = max((spatial_bin - 1) * self.num_orientation_bins, 0)
      endFeatureIdx = min((spatial_bin + 1) * self.num_orientation_bins, len(features))
      workingFeatures = features[startFeatureIdx:endFeatureIdx]
      workingFeatures = cv2.normalize(workingFeatures, workingFeatures, norm_type=cv2.NORM_L2)
      highVals = workingFeatures > self.normalize_threshold
      workingFeatures[highVals] = self.normalize_threshold
      workingFeatures = cv2.normalize(workingFeatures, workingFeatures, norm_type=cv2.NORM_L2)
      normFeatures = np.concatenate((normFeatures, workingFeatures))
    return normFeatures
  
  def process_data(self, data, draw_regions):
    drawing = data.copy()
    features = np.asarray([])
    width = data.shape[1]
    height = data.shape[0]
    cx = int(width/2)
    cy = int(height/2)
    for spatial_bin in range(0, self.num_spatial_bins):
      radius = self.delta_radius * (spatial_bin + 1)
      bins = [0] * self.num_orientation_bins
      for y in range(0, radius):
        lp = self._get_circle_point(y, radius)
        stopPoint = self._get_circle_point(y, radius - self.delta_radius)
        for x in range(stopPoint[0], lp[0] + 1):
          self._bin_values(bins, self._handle_pixel(cx, cy, x, y, data, drawing, draw_regions))
          if x != 0:
            self._bin_values(bins, self._handle_pixel(cx, cy, -x, y, data, drawing, draw_regions))
          if y != 0:
            self._bin_values(bins, self._handle_pixel(cx, cy, x, -y, data, drawing, draw_regions))
          if x != 0 and y != 0:
            self._bin_values(bins, self._handle_pixel(cx, cy, -x, -y, data, drawing, draw_regions))
          if draw_regions and x == stopPoint[0]:
            drawing[cy + y, cx + x] = 0
      features = np.concatenate((features, bins))
    
    if self.normalize:
      features = self._normalize_features(features)

    if draw_regions:
      return features, drawing
    else:
      return features
      

    


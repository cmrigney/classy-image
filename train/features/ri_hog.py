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
  def __init__(self, num_spatial_bins=4, delta_radius=4, num_orientation_bins=13, normalize=True, normalize_block_size=32, normalize_threshold=0.2):
    BaseFeature.__init__(self)
    self.delta_radius = delta_radius
    self.num_orientation_bins = num_orientation_bins
    self.num_spatial_bins = num_spatial_bins
    self.normalize = normalize
    self.normalize_block_size = normalize_block_size
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
    idx = int(idx)
    #need to distribute based on position
    bins[idx] += g
  
  def _vert_interpolate(self, x, y, h, data):
    for i in range(y, y + h):
      temp = data[i, x]
      data[i, x] *= 0.6
      data[i, x] += data[i, x+1]*0.4
      data[i, x+1] *= 0.6
      data[i, x+1] += temp*0.4
  
  def _hor_interpolate(self, x, y, w, data):
    for j in range(x, x + w):
      temp = data[y, j]
      data[y, j] *= 0.6
      data[y, j] += data[y+1, j]*0.4
      data[y+1, j] *= 0.6
      data[y+1, j] += temp*0.4
  
  def _preprocess(self, data):
    result = data.copy()
    width = data.shape[1]
    height = data.shape[0]
    for blockY in range(0, height, int(self.normalize_block_size/2)):
      if blockY + self.normalize_block_size > height:
        blockY = height - self.normalize_block_size
      for blockX in range(0, width, int(self.normalize_block_size/2)):
        if blockX + self.normalize_block_size > width:
          blockX = width - self.normalize_block_size
        p = data[blockY:blockY+self.normalize_block_size, blockX:blockX+self.normalize_block_size]
        val = np.zeros((self.normalize_block_size, self.normalize_block_size))
        val = cv2.normalize(p, val, norm_type=cv2.NORM_L2)
        highVals = val > self.normalize_threshold
        val[highVals] = self.normalize_threshold
        val = cv2.normalize(val, val, norm_type=cv2.NORM_L2)
        result[blockY:blockY+self.normalize_block_size, blockX:blockX+self.normalize_block_size] = val * 255
        if blockX > 0:
          self._vert_interpolate(blockX - 1, blockY, self.normalize_block_size, result)
          self._vert_interpolate(blockX - 2, blockY, self.normalize_block_size, result)
          self._vert_interpolate(blockX, blockY, self.normalize_block_size, result)
        
      if blockY > 0:
        self._hor_interpolate(0, blockY - 1, self.normalize_block_size, result)
        self._hor_interpolate(0, blockY - 2, self.normalize_block_size, result)
        self._hor_interpolate(0, blockY, self.normalize_block_size, result)
    return result

  #this is supposed to be annular cell normalization but does not work right
  def _preprocess2(self, data):
    width = data.shape[1]
    height = data.shape[0]
    cx = int(width/2)
    cy = int(height/2)
    result = data.copy()
    for spatial_bin in range(0, self.num_spatial_bins):
      radius = self.delta_radius * (spatial_bin + 1)
      totals = 0
      for y in range(0, radius):
        lp = self._get_circle_point(y, radius)
        stopPoint = self._get_circle_point(y, radius - self.delta_radius)
        for x in range(stopPoint[0], lp[0] + 1):
          totals += data[cy + y, cx + x]**2
          if x != 0:
            totals += data[cy + y, cx - x]**2
          if y != 0:
            totals += data[cy - y, cx + x]**2
          if x != 0 and y != 0:
            totals += data[cy - y, cx - x]**2
      magnitude = math.sqrt(totals + 2)
      totals = 0

      for y in range(0, radius):
        lp = self._get_circle_point(y, radius)
        stopPoint = self._get_circle_point(y, radius - self.delta_radius)
        for x in range(stopPoint[0], lp[0] + 1):
          totals += _norm_and_clip(cx, cy, x, y, magnitude, result, data)
          if x != 0:
            totals += _norm_and_clip(cx, cy, -x, y, magnitude, result, data)
          if y != 0:
            totals += _norm_and_clip(cx, cy, x, -y, magnitude, result, data)
          if x != 0 and y != 0:
            totals += _norm_and_clip(cx, cy, -x, -y, magnitude, result, data)
      magnitude = math.sqrt(totals + 2)

      for y in range(0, radius):
        lp = self._get_circle_point(y, radius)
        stopPoint = self._get_circle_point(y, radius - self.delta_radius)
        for x in range(stopPoint[0], lp[0] + 1):
          _norm_and_mult(cx, cy, x, y, magnitude, result)
          if x != 0:
            _norm_and_mult(cx, cy, -x, y, magnitude, result)
          if y != 0:
            _norm_and_mult(cx, cy, x, -y, magnitude, result)
          if x != 0 and y != 0:
            _norm_and_mult(cx, cy, -x, -y, magnitude, result)

    return result
      
  
  def process_data(self, data, draw_regions):
    drawing = data.copy()
    features = np.asarray([])
    width = data.shape[1]
    height = data.shape[0]
    cx = int(width/2)
    cy = int(height/2)
    if self.normalize:
      data = self._preprocess(data)
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
    if draw_regions:
      return features, drawing
    else:
      return features
      

    


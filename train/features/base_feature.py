from scipy import misc

class BaseFeature:
  def __init__(self):
    pass
  
  def process_image(self, filename, draw_regions=False):
    data = misc.imread(filename, flatten=True)
    return self.process_data(data, draw_regions)
  
  def process_data(self, data, draw_regions):
    raise Exception("Not implemented")
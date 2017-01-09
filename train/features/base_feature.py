from scipy import misc

class BaseFeature:
  def __init__(self):
    pass
  
  def process_image(filename):
    data = misc.imread(filename)
    return self.process_data(data)
  
  def process_data(data):
    raise Exception("Not implemented")
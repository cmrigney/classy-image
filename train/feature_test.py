from features.circular_histogram_depth_difference import CircularHistogramDepthDifference
import matplotlib.pyplot as plt

c = CircularHistogramDepthDifference(block_size=8, num_rings=3, ring_dist=10, num_blocks_first_ring=6)

features, drawing = c.process_image('test.png', True)

plt.imshow(drawing)
plt.show()

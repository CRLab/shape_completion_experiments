import numpy as np
from mayavi import mlab

mu, sigma = 0, 0.1
x = 10 * np.random.normal(mu, sigma, 5000)
y = 10 * np.random.normal(mu, sigma, 5000)
z = 10 * np.random.normal(mu, sigma, 5000)

weights = np.load('../dropout2layer0.npy')[0].get_value()
weight = weights[3, :, 0, :, :]

# mlab.pipeline.volume(mlab.pipeline.scalar_field(weight))
mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(weight),
                                 vmin=weight.min(), vmax=weight.max())
mlab.outline()
mlab.axes()
mlab.show()

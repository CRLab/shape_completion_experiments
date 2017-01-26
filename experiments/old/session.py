# coding: utf-8

import numpy
import numpy as np
batch_x = np.load('batch_x.npy')
import pcl
x_np = np.zeros((x[0].shape[0], 3))
x = batch_x.nonzero()
x_np = np.zeros((x[0].shape[0], 3))
x_np.shape
x_np[:,2] = x[2]
x_np[:,1] = x[1]
x_np[:,0] = x[0]
x_np.shape
x_np[0]
x_np[1]
x_np[2]
x[2]
len(x)
batch_x.shape
batch_x = batch_x.transpose(0,3,4,1,2)
batch_x.shape
batch_x = batch_x[0,:,:,:,0]
x = batch_x.nonzero()
x_np[:,0] = x[0]
x_np[:,1] = x[1]
x_np[:,2] = x[2]
cloud = pcl.PointCloud(np.array(x_np, np.float32))
pcl.save(cloud, "x.pcd")
get_ipython().magic(u'save session.py 0-28')

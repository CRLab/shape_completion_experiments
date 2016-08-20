#!/usr/bin/python

import numpy as np
import pcl
import sys

infile = sys.argv[-2]
outfile = sys.argv[-1]

data = np.load(infile)
data = data[:,0:3]

pcd = pcl.PointCloud(np.array(data, np.float32))

pcl.save(pcd, outfile)

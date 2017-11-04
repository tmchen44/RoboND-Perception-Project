# Import PCL module
import numpy as np
import pcl
from pcl_helper import *

cloud = pcl.load_XYZRGB('camera_front.pcd')
points_arr = np.asarray(cloud)
mean_arr = np.mean(points_arr, axis=0)[:3]
cent_list = mean_arr.tolist()
print(cent_list)

# Import PCL module
import pcl
from pcl_helper import *

def get_color_list(cluster_count):
    """ Returns a list of randomized colors

        Args:
            cluster_count (int): Number of random colors to generate

        Returns:
            (list): List containing 3-element color lists
    """
    color_list = []
    for i in xrange(cluster_count):
        color_list.append(random_color_gen())
    return color_list

def random_color_gen():
    """ Generates a random color

        Args: None

        Returns:
            list: 3 elements, R, G, and B
    """
    r = randint(0, 255)
    g = randint(0, 255)
    b = randint(0, 255)
    return [r, g, b]

# Load Point Cloud file
cloud = pcl.load_XYZRGB('camera_left.pcd')
outlier_filter = cloud.make_statistical_outlier_filter()
outlier_filter.set_mean_k(20)
x = 0.05
outlier_filter.set_std_dev_mul_thresh(x)
cloud = outlier_filter.filter()

### Voxel Grid filter
vox = cloud.make_voxel_grid_filter()
LEAF_SIZE = 0.01
vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
cloud = vox.filter()

### 1st PassThrough filter in z-axis
passthrough = cloud.make_passthrough_filter()
filter_axis = 'z'
passthrough.set_filter_field_name(filter_axis)
axis_min = 0.4
axis_max = 5.0
passthrough.set_filter_limits(axis_min, axis_max)
cloud = passthrough.filter()

### 2nd PassThrough filter in x-axis
passthrough2 = cloud.make_passthrough_filter()
filter_axis = 'x'
passthrough2.set_filter_field_name(filter_axis)
axis_min = -1.0
axis_max = 0.46
passthrough2.set_filter_limits(axis_min, axis_max)
cloud = passthrough2.filter()

### 3rd PassThrough filter in y-axis
passthrough3 = cloud.make_passthrough_filter()
filter_axis = 'y'
passthrough3.set_filter_field_name(filter_axis)
axis_min = 0.43
axis_max = 5.0
passthrough3.set_filter_limits(axis_min, axis_max)
cloud = passthrough3.filter()

pcl.save(cloud, 'left_passthrough.pcd')

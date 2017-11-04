# Import PCL module
import numpy as np
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

#############################################################################
# Process Front

# Load Point Cloud file
cloud = pcl.load_XYZRGB('camera_front.pcd')
outlier_filter = cloud.make_statistical_outlier_filter()
outlier_filter.set_mean_k(20)
x = 0.05
outlier_filter.set_std_dev_mul_thresh(x)
cloud = outlier_filter.filter()

### Voxel Grid filter
vox = cloud.make_voxel_grid_filter()
LEAF_SIZE = 0.007
vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
cloud = vox.filter()

### PassThrough filter
passthrough = cloud.make_passthrough_filter()
filter_axis = 'z'
passthrough.set_filter_field_name(filter_axis)
axis_min = 0.59
axis_max = 5.0
passthrough.set_filter_limits(axis_min, axis_max)
cloud = passthrough.filter()

### 2nd PassThrough filter in y-axis
passthrough2 = cloud.make_passthrough_filter()
filter_axis = 'y'
passthrough2.set_filter_field_name(filter_axis)
axis_min = -0.55
axis_max = 0.55
passthrough2.set_filter_limits(axis_min, axis_max)
cloud = passthrough2.filter()

### RANSAC plane segmentation
seg = cloud.make_segmenter()
seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)
max_distance = 0.01
seg.set_distance_threshold(max_distance)
inliers, coefficients = seg.segment()

### Extract inliers (tabletop)
extracted_inliers = cloud.extract(inliers, negative=False)

### Extract outliers (objects)
extracted_outliers = cloud.extract(inliers, negative=True)
outlier_filter = extracted_outliers.make_statistical_outlier_filter()
outlier_filter.set_mean_k(20)
x = 0.05
outlier_filter.set_std_dev_mul_thresh(x)
extracted_outliers = outlier_filter.filter()

### Euclidean Clustering
white_cloud = XYZRGB_to_XYZ(extracted_outliers)
tree = white_cloud.make_kdtree()
ec = white_cloud.make_EuclideanClusterExtraction()
ec.set_ClusterTolerance(0.02)
ec.set_MinClusterSize(30)
ec.set_MaxClusterSize(25000)
ec.set_SearchMethod(tree)
cluster_indices = ec.Extract()

# Create Cluster-Mask Point Cloud to visualize each cluster separately
# Assign a color corresponding to each segmented object in scene
cluster_color = get_color_list(len(cluster_indices))
color_cluster_point_list = []
for j, indices in enumerate(cluster_indices):
    for i, indice in enumerate(indices):
        color_cluster_point_list.append([white_cloud[indice][0],
                                        white_cloud[indice][1],
                                        white_cloud[indice][2],
                                         rgb_to_float(cluster_color[j])])

# Create new cloud containing all clusters, each with unique color
cluster_cloud = pcl.PointCloud_PointXYZRGB()
cluster_cloud.from_list(color_cluster_point_list)

filename = 'clustered_objects.pcd'
pcl.save(cluster_cloud, filename)

#############################################################################
# Process Right

# Load Point Cloud file
cloud = pcl.load_XYZRGB('camera_right.pcd')
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
axis_min = -5.0
axis_max = -0.43
passthrough3.set_filter_limits(axis_min, axis_max)
cloud = passthrough3.filter()
cloud = XYZRGB_to_XYZ(cloud)

pcl.save(cloud, 'collision_right.pcd')

#############################################################################
# Process Left

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
cloud = XYZRGB_to_XYZ(cloud)

pcl.save(cloud, 'collision_left.pcd')

#############################################################################
# Combine collision maps into one

front = pcl.load('collision_front.pcd')
right = pcl.load('collision_right.pcd')
left = pcl.load('collision_left.pcd')

combined = pcl.PointCloud()
f = np.asarray(front)
r = np.asarray(right)
l = np.asarray(left)
c = np.concatenate((f, r, l))
combined.from_array(c)

pcl.save(combined, 'collision_combined.pcd')

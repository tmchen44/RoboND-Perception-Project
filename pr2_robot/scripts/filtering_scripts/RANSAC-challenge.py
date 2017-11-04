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
# Cloud class for filtering
class Cloud(object):
    def __init__(self, pcl_cloud):
        self.cloud = pcl_cloud

    # Statistical Outlier Filter
    def stat_outlier_filter(self, mean_k, std_dev):
        outlier_filter = self.cloud.make_statistical_outlier_filter()
        outlier_filter.set_mean_k(mean_k)
        outlier_filter.set_std_dev_mul_thresh(std_dev)
        self.cloud = outlier_filter.filter()

    # Voxel Grid Filter
    def voxel_grid_filter(self, leaf_size):
        vox = self.cloud.make_voxel_grid_filter()
        vox.set_leaf_size(leaf_size, leaf_size, leaf_size)
        self.cloud = vox.filter()

    # Passthrough filter
    def passthrough_filter(self, axis_char, axis_min, axis_max):
        passthrough = self.cloud.make_passthrough_filter()
        passthrough.set_filter_field_name(axis_char)
        passthrough.set_filter_limits(axis_min, axis_max)
        self.cloud = passthrough.filter()

#############################################################################
# Stitch all three clouds together
front_cloud = Cloud(pcl.load_XYZRGB('challenge_front.pcd'))
right_cloud = Cloud(pcl.load_XYZRGB('challenge_right.pcd'))
left_cloud = Cloud(pcl.load_XYZRGB('challenge_left.pcd'))

front_cloud.stat_outlier_filter(mean_k=20, std_dev=0.05)
front_cloud.voxel_grid_filter(leaf_size=0.007)
front_cloud.passthrough_filter('z', 0.5, 5.0)
right_cloud.stat_outlier_filter(mean_k=20, std_dev=0.05)
right_cloud.voxel_grid_filter(leaf_size=0.007)
right_cloud.passthrough_filter('z', 0.5, 5.0)
left_cloud.stat_outlier_filter(mean_k=20, std_dev=0.05)
left_cloud.voxel_grid_filter(leaf_size=0.007)
left_cloud.passthrough_filter('z', 0.5, 5.0)

f = front_cloud.cloud.to_list()
r = right_cloud.cloud.to_list()
l = left_cloud.cloud.to_list()
c = np.concatenate((f, r, l))
combined = pcl.PointCloud_PointXYZRGB()
combined.from_list(c)
pcl.save(combined, 'combined_challenge.pcd')

#############################################################################
# Process Right

# Load Point Cloud file, create Cloud instances
right_cloud = Cloud(pcl.load_XYZRGB('challenge_right.pcd'))

right_cloud.stat_outlier_filter(mean_k=20, std_dev=0.05)
right_cloud.voxel_grid_filter(leaf_size=0.007)

right_cloud_lower = Cloud(right_cloud.cloud)
right_cloud_lower.passthrough_filter('z', 0.56, 5.0)
right_cloud_lower.passthrough_filter('y', -0.7, 0.0)
right_cloud_lower.passthrough_filter('x', -0.5, 0.3)

right_cloud_upper = Cloud(right_cloud.cloud)
right_cloud_upper.passthrough_filter('z', 0.83, 5.0)
right_cloud_upper.stat_outlier_filter(20, 0.05)

#############################################################################
# Process Left

left_cloud = Cloud(pcl.load_XYZRGB('challenge_left.pcd'))

left_cloud.stat_outlier_filter(mean_k=20, std_dev=0.05)
left_cloud.voxel_grid_filter(leaf_size=0.007)

left_cloud_lower = Cloud(left_cloud.cloud)
left_cloud_lower.passthrough_filter('z', 0.56, 0.77)
left_cloud_lower.passthrough_filter('y', 0.0, 0.85)
left_cloud_lower.passthrough_filter('x', -0.5, 0.3)

left_cloud_upper = Cloud(left_cloud.cloud)
left_cloud_upper.passthrough_filter('z', 0.835, 5.0)
left_cloud_upper.stat_outlier_filter(20, 0.05)

all_objects = pcl.PointCloud_PointXYZRGB()
left_l = left_cloud_lower.cloud.to_list()
left_u = left_cloud_upper.cloud.to_list()
right_l = right_cloud_lower.cloud.to_list()
right_u = right_cloud_upper.cloud.to_list()
all_objects.from_list(np.concatenate((left_l, left_u, right_l, right_u)))

pcl.save(all_objects, 'all_challenge_objects.pcd')

#############################################################################
# Euclidean Clustering

white_cloud = XYZRGB_to_XYZ(all_objects)
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

filename = 'clustered_challenge_objects.pcd'
pcl.save(cluster_cloud, filename)

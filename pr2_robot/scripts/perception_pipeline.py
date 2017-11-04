#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
import pcl
import os.path
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import random
import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from std_srvs.srv import Empty
from sensor_msgs.msg import JointState
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml

# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"] = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

def make_output_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    output_dict = {}
    output_dict["test_scene_num"] = test_scene_num
    output_dict["arm_name"] = arm_name
    output_dict["object_name"] = object_name
    output_dict["pick_pose"] = pick_pose
    output_dict["place_pose"] = place_pose
    return output_dict

##############################################################################
# Robot class to handle data

class Robot(object):
    def __init__(self):
        self.state = 'object_detection'
        self.base_angle = None
        self.collision_map = None
        self.detected_objects = []
        self.label_cloud_dict = {} # object label to cloud dictionary
        self.object_list_param = None # ordered list of objects to be picked
        self.output_dict_list = [] # list of params needed in pick-place
        self.num_of_picks = None
        self.picked = 0
        # TODO: Set test scene number here
        self.test_scene = 1

        # ROS node initialization
        rospy.init_node('perception_pipeline', anonymous=True)
        # Create Subscribers
        self.pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, self.make_decision,
                                   queue_size=1)
        self.joint_sub = rospy.Subscriber("/joint_states", JointState, self.update_joint,
                                     queue_size=1)
        # Create Publishers
        self.object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
        self.detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)
        self.collision_map_pub = rospy.Publisher("/pr2/3d_map/points", PointCloud2, queue_size=1)
        # Spin while node is not shutdown
        while not rospy.is_shutdown():
            rospy.spin()

    # Callback function from robot camera Point Cloud Subscriber
    def make_decision(self, pcl_msg):
        if self.state == 'object_detection':
            self.detect_objects(pcl_msg)
            self.state = 'create_pick_list'
        elif self.state == 'create_pick_list':
            self.create_pick_list()
            self.rotate(-np.pi/2)
            self.state = 'flush_right'
        elif self.state == 'flush_right':
            # this state flushes the unneeded point cloud from the queue
            self.state = 'update_collision_map_right'
        elif self.state == 'update_collision_map_right':
            self.update_collision_map(pcl_msg)
            self.rotate(np.pi/2)
            self.state = 'flush_left'
        elif self.state == 'flush_left':
            # this state flushes the unneeded point cloud from the queue
            self.state = 'update_collision_map_left'
        elif self.state == 'update_collision_map_left':
            self.update_collision_map(pcl_msg)
            self.rotate(0.0)
            self.state = 'publish_collision_map'
        elif self.state == 'publish_collision_map':
            self.publish_collision_map()
            self.state = 'pick_objects'
        elif self.state == 'pick_objects':
            self.pr2_mover()
            if self.picked < self.num_of_picks:
                self.state = 'publish_collision_map'
            else:
                self.state = 'finished'
        elif self.state == 'finished':
            pass

    # Callback function for base joint angle subscriber
    def update_joint(self, joint_msg):
        self.base_angle = joint_msg.position[19]

    def detect_objects(self, pcl_msg):
        """
        This method takes in the scene in front of the robot, carries out
        object detection, and generates the front part of the collision map
        """
        # Convert ROS msg to PCL data
        cloud = ros_to_pcl(pcl_msg)
        # Statistical Outlier Filtering
        outlier_filter = cloud.make_statistical_outlier_filter()
        outlier_filter.set_mean_k(20)
        x = 0.05
        outlier_filter.set_std_dev_mul_thresh(x)
        cloud = outlier_filter.filter()
        # Voxel Grid Downsampling
        vox = cloud.make_voxel_grid_filter()
        LEAF_SIZE = 0.007
        vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
        cloud = vox.filter()
        # 1st PassThrough filter in z-axis
        passthrough = cloud.make_passthrough_filter()
        filter_axis = 'z'
        passthrough.set_filter_field_name(filter_axis)
        axis_min = 0.59
        axis_max = 5.0
        passthrough.set_filter_limits(axis_min, axis_max)
        cloud = passthrough.filter()
        # 2nd PassThrough filter in y-axis
        passthrough2 = cloud.make_passthrough_filter()
        filter_axis = 'y'
        passthrough2.set_filter_field_name(filter_axis)
        axis_min = -0.55
        axis_max = 0.55
        passthrough2.set_filter_limits(axis_min, axis_max)
        cloud = passthrough2.filter()
        # RANSAC Plane Segmentation
        seg = cloud.make_segmenter()
        seg.set_model_type(pcl.SACMODEL_PLANE)
        seg.set_method_type(pcl.SAC_RANSAC)
        max_distance = 0.01
        seg.set_distance_threshold(max_distance)
        inliers, coefficients = seg.segment()
        # Extract inliers and outliers as subset point clouds
        self.collision_map = cloud.extract(inliers, negative=False)
        extracted_outliers = cloud.extract(inliers, negative=True)
        outlier_filter = extracted_outliers.make_statistical_outlier_filter()
        outlier_filter.set_mean_k(20)
        x = 0.05
        outlier_filter.set_std_dev_mul_thresh(x)
        extracted_outliers = outlier_filter.filter()
        # Euclidean Clustering
        white_cloud = XYZRGB_to_XYZ(extracted_outliers)
        tree = white_cloud.make_kdtree()
        ec = white_cloud.make_EuclideanClusterExtraction()
        ec.set_ClusterTolerance(0.02)
        ec.set_MinClusterSize(30)
        ec.set_MaxClusterSize(25000)
        ec.set_SearchMethod(tree)
        cluster_indices = ec.Extract()
        # Classify the clusters! (loop through each detected cluster one at a time)
        detected_objects_labels = []
        # Grab the indices of extracted_outliers/white_cloud for each cluster
        for index, pts_list in enumerate(cluster_indices):
            # Extract the points for the current cluster using indices (pts_list)
            pcl_cluster = extracted_outliers.extract(pts_list)
            # Compute the associated feature vector
            ros_cluster = pcl_to_ros(pcl_cluster)
            chists = compute_color_histograms(ros_cluster, using_hsv=True)
            normals = get_normals(ros_cluster)
            nhists = compute_normal_histograms(normals)
            feature = np.concatenate((chists, nhists))
            # Make the prediction
            prediction = clf.predict(scaler.transform(feature.reshape(1, -1)))
            label = encoder.inverse_transform(prediction)[0]
            detected_objects_labels.append(label)
            # Publish a label into RViz, place above 1st cluster point
            label_pos = list(white_cloud[pts_list[0]])
            label_pos[2] += 0.4
            self.object_markers_pub.publish(make_label(label, label_pos, index))
            # Add the detected object to the list of detected objects.
            do = DetectedObject()
            do.label = label
            do.cloud = ros_cluster
            self.detected_objects.append(do)
        print 'Detected {} objects: {}'.format(len(detected_objects_labels),
    					                           detected_objects_labels)
        # Publish the list of detected objects
        self.detected_objects_pub.publish(self.detected_objects)

    # Function to generate list of params needed for pick-place operations
    def create_pick_list(self):
        # Create label-centroid and label-cloud dictionaries from object_list
        label_centroid_dict = {}
        for obj in self.detected_objects:
            pcl_cloud = ros_to_pcl(obj.cloud)
            points_arr = np.asarray(pcl_cloud)
            numpy_centroids = np.mean(points_arr, axis=0)[:3]
            label_centroid_dict[obj.label] = numpy_centroids.tolist()
            self.label_cloud_dict[obj.label] = pcl_cloud
        # Read parameters from current pick_list
        self.object_list_param = rospy.get_param('/object_list')
        self.num_of_picks = len(self.object_list_param)
        # Set test_scene and dropbox centroids
        test_scene_num = Int32()
        test_scene_num.data = self.test_scene
        dropbox_poses = {}
        dropbox_param = rospy.get_param('/dropbox')
        dropbox_poses['left'] = dropbox_param[0]['position']
        dropbox_poses['right'] = dropbox_param[1]['position']
        # Loop through the pick list
        yaml_dict_list = []
        for i in range(self.num_of_picks):
            # Obtain object name and group
            object_name = String()
            object_name.data = self.object_list_param[i]['name']
            object_group = self.object_list_param[i]['group']
            # Obtain centroid of object, assign to pick_pose
            centroid = label_centroid_dict[object_name.data]
            pick_pose = Pose()
            pick_pose.position.x = centroid[0]
            pick_pose.position.y = centroid[1]
            pick_pose.position.z = centroid[2]
            # Assign the arm to be used for pick_place
            arm_name = String()
            if object_group == 'green':
                arm_name.data = 'right'
            else:
                arm_name.data = 'left'
            # Create 'place_pose' for the object
            place_pose = Pose()
            # Randomize place position
            rand_x = random.randint(-10, -3) * 0.01
            rand_y = random.randint(-8, 8) * 0.01
            place_pose.position.x = dropbox_poses[arm_name.data][0] + rand_x
            place_pose.position.y = dropbox_poses[arm_name.data][1] + rand_y
            place_pose.position.z = dropbox_poses[arm_name.data][2]
            # Create list of dictionaries to store data for use in pick and place
            output_dict_entry = make_output_dict(test_scene_num, arm_name, object_name,
                                             pick_pose, place_pose)
            self.output_dict_list.append(output_dict_entry)
            # Create a list of dictionaries (made with make_yaml_dict())
            # for later output to yaml format
            yaml_dict_entry = make_yaml_dict(test_scene_num, arm_name, object_name,
                                             pick_pose, place_pose)
            yaml_dict_list.append(yaml_dict_entry)

        # Output your request parameters into output yaml file
        send_to_yaml('output.yaml', yaml_dict_list)

    def rotate(self, angle):
        # initialize joint control publisher
        pub_j1 = rospy.Publisher('/pr2/world_joint_controller/command', Float64,
                                 queue_size=10)
        # rotate robot to desired angle
        while not self.at_goal(angle):
            pub_j1.publish(angle)

    def at_goal(self, goal):
        tolerance = 0.01
        return abs(self.base_angle - goal) <= tolerance

    def update_collision_map(self, pcl_msg):
        # Convert ROS msg to PCL data
        cloud = ros_to_pcl(pcl_msg)
        # Statistical outlier filter
        outlier_filter = cloud.make_statistical_outlier_filter()
        outlier_filter.set_mean_k(20)
        x = 0.05
        outlier_filter.set_std_dev_mul_thresh(x)
        cloud = outlier_filter.filter()
        # Voxel Grid filter
        vox = cloud.make_voxel_grid_filter()
        LEAF_SIZE = 0.01
        vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
        cloud = vox.filter()
        # 1st PassThrough filter in z-axis
        passthrough = cloud.make_passthrough_filter()
        filter_axis = 'z'
        passthrough.set_filter_field_name(filter_axis)
        axis_min = 0.4
        axis_max = 5.0
        passthrough.set_filter_limits(axis_min, axis_max)
        cloud = passthrough.filter()
        # 2nd PassThrough filter in x-axis
        passthrough2 = cloud.make_passthrough_filter()
        filter_axis = 'x'
        passthrough2.set_filter_field_name(filter_axis)
        axis_min = -1.0
        axis_max = 0.46
        passthrough2.set_filter_limits(axis_min, axis_max)
        cloud = passthrough2.filter()
        # 3rd PassThrough filter in y-axis
        passthrough3 = cloud.make_passthrough_filter()
        filter_axis = 'y'
        passthrough3.set_filter_field_name(filter_axis)
        if self.state == 'update_collision_map_right':
            axis_min = -5.0
            axis_max = -0.43
        elif self.state == 'update_collision_map_left':
            axis_min = 0.43
            axis_max = 5.0
        passthrough3.set_filter_limits(axis_min, axis_max)
        cloud = passthrough3.filter()
        # Update collision map
        combined = pcl.PointCloud_PointXYZRGB()
        existing = np.asarray(self.collision_map)
        addendum = np.asarray(cloud)
        updated = np.concatenate((existing, addendum))
        combined.from_array(updated)
        self.collision_map = combined

    # Function to generate appropriate collision map for PickPlace Service
    def publish_collision_map(self):
        # Clear octomap
        rospy.wait_for_service('/clear_octomap')
        clear_octomap = rospy.ServiceProxy('/clear_octomap', Empty)
        try:
            resp = clear_octomap()
        except rospy.ServiceException as e:
            print("Service did not process request: " + str(e))
        # Append remaining table objects to collision map and publish
        base_array = np.asarray(self.collision_map)
        for i in range(self.picked + 1, self.num_of_picks):
            obj_label = self.object_list_param[i]['name']
            obj_array = np.asarray(self.label_cloud_dict[obj_label])
            base_array = np.concatenate((base_array, obj_array))
        combined_cloud = pcl.PointCloud_PointXYZRGB()
        combined_cloud.from_array(base_array)
        self.collision_map_pub.publish(pcl_to_ros(combined_cloud))

    # Function to load parameters and request PickPlace service
    def pr2_mover(self):
        # Load parameters for current object
        curr_obj_dict = self.output_dict_list[self.picked]
        test_scene_num = curr_obj_dict["test_scene_num"]
        object_name = curr_obj_dict["object_name"]
        arm_name = curr_obj_dict["arm_name"]
        pick_pose = curr_obj_dict["pick_pose"]
        place_pose = curr_obj_dict["place_pose"]
        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')
        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)
            # Insert your message variables to be sent as a service request
            resp = pick_place_routine(test_scene_num, object_name, arm_name,
                                      pick_pose, place_pose)
            print("Response: ", resp.success)
        except rospy.ServiceException, e:
            print "Service call failed: %s" %e
        # Increment target object counter
        self.picked += 1

# End Robot class
###############################################################################

if __name__ == '__main__':
    # Load classifier model from disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']
    # Initialize color_list
    get_color_list.color_list = []
    # Create robot instance
    Robot()

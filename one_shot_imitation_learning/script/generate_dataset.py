#!/usr/bin/env python3

import rospy, rospkg, os
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
from gazebo_msgs.srv import GetModelState
from one_shot_imitation_learning.srv import GetPose
from cv_bridge import CvBridge
import tf.transformations as tf
import cv2
import numpy as np


class DatasetGenerator:

    TOLERENCE = 0.01
    STEP_SIZE = 0.005

    def __init__(self, trajectory_name, model_name):
        self.index = 0
        self.DATASET_PATH = rospkg.RosPack().get_path('one_shot_imitation_learning') + '/dataset/' + trajectory_name + '/'
        if not os.path.exists(self.DATASET_PATH):
            os.makedirs(self.DATASET_PATH)
        self.model_name = model_name
        self.cv_bridge = CvBridge()
        self.labels = np.empty([0, 6])

        rospy.wait_for_service('/gazebo/get_model_state')
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        rospy.wait_for_service('get_gripper_pose')
        self.get_gripper_pose = rospy.ServiceProxy('get_gripper_pose', GetPose)
        self.cartesian_pose_pub = rospy.Publisher('pose_displacement', Pose, queue_size=10)
        self.camera_sub = rospy.Subscriber('camera/image_raw', Image, self.callback, queue_size=10)
    
    def callback(self, image):

        response = self.get_model_state(self.model_name, 'world')
        model_pose = response.pose

        response = self.get_gripper_pose()
        gripper_pose = response.pose

        if (gripper_pose.position.x - model_pose.position.x) > DatasetGenerator.TOLERENCE:
            ex = -DatasetGenerator.STEP_SIZE
        elif (gripper_pose.position.x - model_pose.position.x) < -DatasetGenerator.TOLERENCE:
            ex = DatasetGenerator.STEP_SIZE
        else:
            ex = 0

        if (gripper_pose.position.y - model_pose.position.y) > DatasetGenerator.TOLERENCE:
            ey = -DatasetGenerator.STEP_SIZE
        elif (gripper_pose.position.y - model_pose.position.y) < -DatasetGenerator.TOLERENCE:
            ey = DatasetGenerator.STEP_SIZE
        else:
            ey = 0

        if (gripper_pose.position.z - 0.35) > DatasetGenerator.TOLERENCE:
            ez = -DatasetGenerator.STEP_SIZE
        elif (gripper_pose.position.z - 0.35) < -DatasetGenerator.TOLERENCE:
            ez = DatasetGenerator.STEP_SIZE
        else:
            ez = 0

        _, _, gripper_angle = tf.euler_from_quaternion([gripper_pose.orientation.x, gripper_pose.orientation.y, gripper_pose.orientation.z, gripper_pose.orientation.w])
        _, _, model_angle = tf.euler_from_quaternion([model_pose.orientation.x, model_pose.orientation.y, model_pose.orientation.z, model_pose.orientation.w])

        if (gripper_angle - model_angle - 1.5708) > DatasetGenerator.TOLERENCE:
            er = -DatasetGenerator.STEP_SIZE
        elif (gripper_angle - model_angle - 1.5708) < -DatasetGenerator.TOLERENCE:
            er = DatasetGenerator.STEP_SIZE
        else:
            er = 0
        
        if (ex == 0) and (ey == 0) and (ez == 0) and (er == 0):
            rospy.loginfo('Reached Goal Pose')
            return
        
        er_cos = np.cos(er)
        er_sin = np.sin(er)
        print(gripper_angle - 1.5708, model_angle)
        print(f'{ex}, {ey}, {ez}, {er}, {er_cos}, {er_sin}')

        # Move Gripper
        self.move_gripper(ex, ey, ez, er_cos, er_sin)

        # Get Image
        cv2_image = self.cv_bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
        image_name = str(self.index) + '.png'
        cv2.imwrite(self.DATASET_PATH + image_name, cv2_image)

        labels_name_full_path = self.DATASET_PATH + 'labels.csv'
        self.labels = np.vstack([self.labels, [image_name, ex, ey, ez, er_cos, er_sin]])
        np.savetxt(labels_name_full_path, self.labels.astype(str), delimiter=',', fmt='%s')
        rospy.loginfo(f'Save image: {image_name}, label index: {self.index}')

        self.index += 1

    def move_gripper(self, ex, ey, ez, er_cos, er_sin):
        displacement = Pose()
        displacement.position.x = ex
        displacement.position.y = ey
        displacement.position.z = ez

        er = np.arctan2(er_sin, er_cos)
        print(f'move angle: {er}')
        eq = tf.quaternion_from_euler(0, 0, er)
        displacement.orientation.x = eq[0]
        displacement.orientation.y = eq[1]
        displacement.orientation.z = eq[2]
        displacement.orientation.w = eq[3]

        self.cartesian_pose_pub.publish(displacement)
        


if __name__ == '__main__':
    rospy.init_node('generate_dataset')

    trajectory_name = input('Trajectory_name: ')
    model_name = 'Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White'

    dataset_generator = DatasetGenerator(trajectory_name, model_name)

    # cartesian_pose_pub = rospy.Publisher('pose_displacement', Pose, queue_size=10)

    # new_pose = Pose()
    # new_pose.position.x = -0.1
    # print(new_pose)
    # rospy.sleep(1)
    # cartesian_pose_pub.publish(new_pose)
    rospy.spin()
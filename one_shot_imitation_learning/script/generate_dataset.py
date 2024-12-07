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
    KP_XYZ = 0.3
    KP_R = 0.3
    MAX_STEP_SIZE = 0.05   # Maximum movement stepsize

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

        ex = model_pose.position.x - gripper_pose.position.x
        if abs(ex) > DatasetGenerator.TOLERENCE:
            ex *= DatasetGenerator.KP_XYZ
            ex = round(ex, 4)
            if ex > DatasetGenerator.MAX_STEP_SIZE:
                ex = DatasetGenerator.MAX_STEP_SIZE
            elif ex < -DatasetGenerator.MAX_STEP_SIZE:
                ex = -DatasetGenerator.MAX_STEP_SIZE
        else:
            ex = 0
        
        ey = model_pose.position.y - gripper_pose.position.y
        print(ey)
        if abs(ey) > DatasetGenerator.TOLERENCE:
            ey *= DatasetGenerator.KP_XYZ
            ey = round(ey, 4)
            if ey > DatasetGenerator.MAX_STEP_SIZE:
                ey = DatasetGenerator.MAX_STEP_SIZE
            elif ey < -DatasetGenerator.MAX_STEP_SIZE:
                ey = -DatasetGenerator.MAX_STEP_SIZE
        else:
            ey = 0

        ez = 0.35 - gripper_pose.position.z
        if abs(ez) > DatasetGenerator.TOLERENCE:
            ez *= DatasetGenerator.KP_XYZ
            ez = round(ez, 4)
            if ez > DatasetGenerator.MAX_STEP_SIZE:
                ez = DatasetGenerator.MAX_STEP_SIZE
            elif ez < -DatasetGenerator.MAX_STEP_SIZE:
                ez = -DatasetGenerator.MAX_STEP_SIZE
        else:
            ez = 0

        _, _, gripper_angle = tf.euler_from_quaternion([gripper_pose.orientation.x, gripper_pose.orientation.y, gripper_pose.orientation.z, gripper_pose.orientation.w])
        _, _, model_angle = tf.euler_from_quaternion([model_pose.orientation.x, model_pose.orientation.y, model_pose.orientation.z, model_pose.orientation.w])

        er = model_angle - gripper_angle + 1.5708
        if abs(er) > DatasetGenerator.TOLERENCE:
            er *= DatasetGenerator.KP_R
            if er > DatasetGenerator.MAX_STEP_SIZE:
                er = DatasetGenerator.MAX_STEP_SIZE
            elif er < -DatasetGenerator.MAX_STEP_SIZE:
                er = -DatasetGenerator.MAX_STEP_SIZE
        else:
            er = 0

        er_cos = np.cos(er)
        er_cos = round(er_cos, 4)
        er_sin = np.sin(er)
        er_sin = round(er_sin, 4)
        
        if (ex == 0) and (ey == 0) and (ez == 0) and (er == 0):
            rospy.loginfo('Reached Goal Pose')
            # Get Image
            cv2_image = self.cv_bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
            image_name = str(self.index) + '.png'
            cv2.imwrite(self.DATASET_PATH + image_name, cv2_image)

            labels_name_full_path = self.DATASET_PATH + 'labels.csv'
            self.labels = np.vstack([self.labels, [image_name, ex, ey, ez, er_cos, er_sin]])
            np.savetxt(labels_name_full_path, self.labels.astype(str), delimiter=',', fmt='%s')
            rospy.loginfo(f'Save image: {image_name}, label index: {self.index}')
            rospy.signal_shutdown('Reached Goal Pose')
            return
        
        rospy.loginfo(f'ex: {ex}, ey: {ey}, ez: {ez}, er: {er}, er_cos: {er_cos}, er_sin: {er_sin}')

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
        rospy.loginfo(f'Moving Angle: {er}')
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
    rospy.spin()
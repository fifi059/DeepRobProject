#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import Pose
import tf.transformations as tf

def move_gripper(ex, ey, ez, er_cos, er_sin, publisher):
        displacement = Pose()
        displacement.position.x = ex
        displacement.position.y = ey
        displacement.position.z = ez

        er = np.arctan2(er_sin, er_cos)
        print(f'Moving Angle: {er}')
        eq = tf.quaternion_from_euler(0, 0, er)
        displacement.orientation.x = eq[0]
        displacement.orientation.y = eq[1]
        displacement.orientation.z = eq[2]
        displacement.orientation.w = eq[3]

        publisher.publish(displacement)

if __name__ == '__main__':
    rospy.init_node('test_dataset')

    cartesian_pose_pub = rospy.Publisher('pose_displacement', Pose, queue_size=10)
    rospy.sleep(2)

    actions_path = '/root/deeprob_project_ws/src/DeepRobProject/one_shot_imitation_learning/script/Test_001_test_results_combined.csv'
    actions = np.loadtxt(actions_path, delimiter=',')

    for i in range(actions.shape[0]):
          ex = actions[i][0]
          ey = actions[i][1]
          ez = actions[i][2]
          er_cos = actions[i][3]
          er_sin = actions[i][4]
          print(ex, ey, ez, er_cos, er_sin)
          move_gripper(ex, ey, ez, er_cos, er_sin, cartesian_pose_pub)
          rospy.sleep(5)


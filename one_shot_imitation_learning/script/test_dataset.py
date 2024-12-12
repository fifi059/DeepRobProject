#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import Pose
import tf.transformations as tf
from gazebo_msgs.srv import GetModelState
from one_shot_imitation_learning.srv import GetPose

# def move_gripper(ex, ey, ez, er_cos, er_sin, publisher):
#         displacement = Pose()
#         displacement.position.x = ex
#         displacement.position.y = ey
#         displacement.position.z = ez

#         er = np.arctan2(er_sin, er_cos)
#         print(f'Moving Angle: {er}')
#         eq = tf.quaternion_from_euler(0, 0, er)
#         displacement.orientation.x = eq[0]
#         displacement.orientation.y = eq[1]
#         displacement.orientation.z = eq[2]
#         displacement.orientation.w = eq[3]

#         publisher.publish(displacement)

reach_goal_count = 0
def move_gripper(ex, ey, ez, er_cos, er_sin, publisher, get_model_state, get_gripper_pose):

        # Filter the output
        if abs(ex) > 0.01:
                if ex > 0.1:
                        ex = 0.1
                elif ex < -0.1:
                        ex = -0.1
        else:
                ex = 0

        if abs(ey) > 0.01:
                if ey > 0.1:
                        ey = 0.1
                elif ey < -0.1:
                        ey = -0.1
        else:
                ey = 0

        if abs(ez) > 0.01:
                if ez > 0.1:
                        ez = 0.1
                elif ez < -0.1:
                        ez = -0.1
        else:
                ez = 0

        er = np.arctan2(er_sin, er_cos)
        if abs(er) > 0.01:
                if er > 0.2:
                        er = 0.2
                elif er < -0.2:
                        er = -0.2
        else:
                er = 0
        # rospy.loginfo(f'Moving Angle: {er}')
        eq = tf.quaternion_from_euler(0, 0, er)

        displacement = Pose()
        displacement.position.x = ex
        displacement.position.y = ey
        displacement.position.z = ez

        displacement.orientation.x = eq[0]
        displacement.orientation.y = eq[1]
        displacement.orientation.z = eq[2]
        displacement.orientation.w = eq[3]

        if (ex == 0) and (ey == 0) and (ez == 0) and (er == 0):
                rospy.loginfo('Reached Goal Pose')
                reach_goal_count = 5
        else:
                reach_goal_count = 0
                print(f'ex: {ex}, ey: {ey}, ez: {ez}, er: {er}')
                publisher.publish(displacement)

        if reach_goal_count >= 5:
                response = get_model_state('Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White', 'world')
                model_pose = response.pose

                response = get_gripper_pose()
                gripper_pose = response.pose

                ex = abs(model_pose.position.x - gripper_pose.position.x)
                ey = abs(model_pose.position.y - gripper_pose.position.y)
                ez = abs(0.35 - gripper_pose.position.z)
                _, _, gripper_angle = tf.euler_from_quaternion([gripper_pose.orientation.x, gripper_pose.orientation.y, gripper_pose.orientation.z, gripper_pose.orientation.w])
                _, _, model_angle = tf.euler_from_quaternion([model_pose.orientation.x, model_pose.orientation.y, model_pose.orientation.z, model_pose.orientation.w])

                er = abs(model_angle - gripper_angle + 1.5708)

                if (ex < 0.05) and (ey < 0.05) and (ez < 0.05):
                        rospy.loginfo(f'Translation Success! ex: {ex}, ey: {ey}, ez: {ez}')
                else:
                        rospy.loginfo(f'Translation Fail: ex: {ex}, ey: {ey}, ez: {ez}')

                if er < 0.05:
                        rospy.loginfo(f'Rotation Success! er:{er}')
                else:
                        rospy.loginfo(f'Rotation Fail: er:{er}')
                
                # rospy.signal_shutdown('Reached Goal Pose')

if __name__ == '__main__':
    rospy.init_node('test_dataset')

    cartesian_pose_pub = rospy.Publisher('pose_displacement', Pose, queue_size=10)
    rospy.wait_for_service('/gazebo/get_model_state')
    get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    rospy.wait_for_service('get_gripper_pose')
    get_gripper_pose = rospy.ServiceProxy('get_gripper_pose', GetPose)
    rospy.sleep(2)

    actions_path = '/root/deeprob_project_ws/src/DeepRobProject/one_shot_imitation_learning/script/new_test_results_combined.csv'
#     actions = np.loadtxt(actions_path, delimiter=',', usecols=range(1,6))
    actions = np.loadtxt(actions_path, delimiter=',', skiprows=1)

    for i in range(actions.shape[0]):
          ex = actions[i][0]
          ey = actions[i][1]
          ez = actions[i][2]
          er_cos = actions[i][3]
          er_sin = actions[i][4]
          print(f'ex:{ex},ey:{ey},ez:{ez},er_cos:{er_cos},er_sin:{er_sin}')
          move_gripper(ex, ey, ez, er_cos, er_sin, cartesian_pose_pub, get_model_state, get_gripper_pose)
          rospy.sleep(2)


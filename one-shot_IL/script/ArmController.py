#!/usr/bin/env python3

import rospy, sys
import moveit_commander
import moveit_msgs.msg
from math import pi
import tf.transformations as tf


class ArmController:

    def __init__(self):
        
        # Initialize the node
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('arm_controller')

        try:
            self.is_gripper_present = rospy.get_param(rospy.get_namespace() + "is_gripper_present", False)
            if self.is_gripper_present:
                gripper_joint_names = rospy.get_param(rospy.get_namespace() + "gripper_joint_names", [])
                self.gripper_joint_name = gripper_joint_names[0]
            else:
                self.gripper_joint_name = ""
            self.degrees_of_freedom = rospy.get_param(rospy.get_namespace() + "degrees_of_freedom", 7)

            # Create the MoveItInterface necessary objects
            arm_group_name = "arm"
            self.robot = moveit_commander.RobotCommander("robot_description")
            self.scene = moveit_commander.PlanningSceneInterface(ns=rospy.get_namespace())
            self.arm_group = moveit_commander.MoveGroupCommander(arm_group_name, ns=rospy.get_namespace())
            self.display_trajectory_publisher = rospy.Publisher(rospy.get_namespace() + 'move_group/display_planned_path',
                                                            moveit_msgs.msg.DisplayTrajectory,
                                                            queue_size=20)

            if self.is_gripper_present:
                gripper_group_name = "gripper"
                self.gripper_group = moveit_commander.MoveGroupCommander(gripper_group_name, ns=rospy.get_namespace())

            rospy.loginfo("Initializing node in namespace " + rospy.get_namespace())
        except Exception as e:
            print (e)
            self.is_init_success = False
        else:
            self.is_init_success = True

    def get_cartesian_pose(self):
        arm_group = self.arm_group

        # Get the current pose and display it
        pose = arm_group.get_current_pose()   # type -> PoseStamped, contains header + pose
        rospy.loginfo("Actual cartesian pose is : ")
        rospy.loginfo(pose.pose)

        return pose.pose

    def reach_cartesian_pose(self, pose, tolerance, constraints):
        arm_group = self.arm_group
        
        # Set the tolerance
        arm_group.set_goal_position_tolerance(tolerance)

        # Set the trajectory constraint if one is specified
        if constraints is not None:
            arm_group.set_path_constraints(constraints)

        # Get the current Cartesian Position
        arm_group.set_pose_target(pose)

        # Plan and execute
        rospy.loginfo("Planning and going to the Cartesian Pose")
        return arm_group.go(wait=True)
    

if __name__ == '__main__':
    arm_controller = ArmController()

    success = arm_controller.is_init_success
    
    if success:
        rospy.loginfo("Reaching Cartesian Pose...")
        
        actual_pose = arm_controller.get_cartesian_pose()
        angle_radians = 90 * (pi / 180)
        rotation_quat = tf.quaternion_about_axis(angle_radians, (0, 1, 0))

        current_quat = [
            actual_pose.orientation.x,
            actual_pose.orientation.y,
            actual_pose.orientation.z,
            actual_pose.orientation.w,
        ]
        new_orientation = tf.quaternion_multiply(rotation_quat, current_quat)
        actual_pose.orientation.x = new_orientation[0]
        actual_pose.orientation.y = new_orientation[1]
        actual_pose.orientation.z = new_orientation[2]
        actual_pose.orientation.w = new_orientation[3]

        # actual_pose.position.x += 0.02
        # actual_pose.position.y -= 0.01
        success &= arm_controller.reach_cartesian_pose(pose=actual_pose, tolerance=0.01, constraints=None)
        print (success)
    rospy.spin()
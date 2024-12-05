#!/usr/bin/env python3

import rospy, sys
import moveit_commander
import moveit_msgs.msg
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Pose
from math import pi
import tf.transformations as tf
from one_shot_imitation_learning.srv import GetPose, GetPoseResponse


class ArmController:

    HOME_ACTION_IDENTIFIER = 2

    def __init__(self):

        rospy.init_node('arm_controller')
        
        # Check if the environment has been initialized
        gazebo_is_initialized = False
        while gazebo_is_initialized is False:
            gazebo_is_initialized = rospy.get_param(rospy.get_namespace() + 'gazebo_is_initialized', False)
        rospy.loginfo('The environment has been initialized. Begin Initialize Arm Controller!')
        
        # Initialize the node
        moveit_commander.roscpp_initialize(sys.argv)
        self.joint_angles_sub = rospy.Subscriber('angle_displacement', Float32MultiArray, self.joint_angles_callback, queue_size=10)
        self.cartesian_pose_sub = rospy.Subscriber('pose_displacement', Pose, self.cartesian_pose_callback, queue_size=10)
        self.cartesian_pose_service = rospy.Service('get_gripper_pose', GetPose, self.handle_get_pose)

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
    
        # From Home Pose to Pregrasp Pose
        if self.is_init_success:
        
            rospy.loginfo("Reaching Pregrasp Pose...")
            current_pose = self.get_cartesian_pose()
            angle_radians = 90 * (pi / 180)
            rotation_quat = tf.quaternion_about_axis(angle_radians, (0, 1, 0))

            current_quat = [
                current_pose.orientation.x,
                current_pose.orientation.y,
                current_pose.orientation.z,
                current_pose.orientation.w,
            ]
            new_orientation = tf.quaternion_multiply(rotation_quat, current_quat)

            new_pose = current_pose
            new_pose.position.x -= 0.3
            new_pose.orientation.x = new_orientation[0]
            new_pose.orientation.y = new_orientation[1]
            new_pose.orientation.z = new_orientation[2]
            new_pose.orientation.w = new_orientation[3]

            result = self.reach_cartesian_pose(new_pose)

            if result is True:
                rospy.loginfo("Arm Controller init Complete!")

                self.pregrasp = self.get_joint_angles()

    def handle_get_pose(self, req):

        current_pose = self.get_cartesian_pose()
        return GetPoseResponse(current_pose)

    def joint_angles_callback(self, angles):
        
        current_angles = self.get_joint_angles()

        new_angles = current_angles
        for i in range(new_angles):
            new_angles[i] += angles[i]
        
        result = self.reach_joint_angles(new_angles)
        if result is True:
            rospy.loginfo("Successfully reach new angles")
        else:
            rospy.loginfo("Cannot reach new angles")
        return

    def cartesian_pose_callback(self, pose):
        
        current_pose = self.get_cartesian_pose()

        new_pose = current_pose
        new_pose.position.x += pose.position.x
        new_pose.position.y += pose.position.y
        new_pose.position.z += pose.position.z
        
        rotation_quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        current_quat = [current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w]
        new_orientation = tf.quaternion_multiply(rotation_quat, current_quat)
        new_pose.orientation.x = new_orientation[0]
        new_pose.orientation.y = new_orientation[1]
        new_pose.orientation.z = new_orientation[2]
        new_pose.orientation.w = new_orientation[3]

        result = self.reach_cartesian_pose(new_pose)

        if result is True:
            rospy.loginfo("Successfully reach new pose")
        else:
            rospy.loginfo("Cannot reach new pose")
        return

    def reach_pregrasp(self):

        rospy.loginfo("Reaching Pregrasp Pose...")
        return self.reach_joint_angles(self.pregrasp)

    def get_joint_angles(self):
        arm_group = self.arm_group

        # Get the current pose and display it
        joint_angles = arm_group.get_current_joint_values()
        # rospy.loginfo("Current joint angles are : ")
        # rospy.loginfo(joint_angles)

        return joint_angles
    
    def reach_joint_angles(self, joint_angles: list, tolerance=0.001):
        arm_group = self.arm_group

        # Set the goal joint tolerance
        self.arm_group.set_goal_joint_tolerance(tolerance)

        # Set the current Joint Angles
        arm_group.set_joint_value_target(joint_angles)

        # Plan and execute
        rospy.loginfo("Planning and going to the Joint Angles")
        return arm_group.go(wait=True)

    def get_cartesian_pose(self):
        arm_group = self.arm_group

        # Get the current pose and display it
        pose = arm_group.get_current_pose()   # type -> PoseStamped, contains header + pose
        # rospy.loginfo("Current cartesian pose is : ")
        # rospy.loginfo(pose.pose)

        return pose.pose

    def reach_cartesian_pose(self, pose, tolerance=0.001, constraints=None):
        arm_group = self.arm_group
        
        # Set the tolerance
        arm_group.set_goal_position_tolerance(tolerance)

        # Set the trajectory constraint if one is specified
        if constraints is not None:
            arm_group.set_path_constraints(constraints)

        # Set the current Cartesian Position
        arm_group.set_pose_target(pose)

        # Plan and execute
        rospy.loginfo("Planning and going to the Cartesian Pose")
        return arm_group.go(wait=True)
    
    def reach_gripper_position(self, relative_position):
        gripper_group = self.gripper_group
        
        # We only have to move this joint because all others are mimic!
        gripper_joint = self.robot.get_joint(self.gripper_joint_name)
        gripper_max_absolute_pos = gripper_joint.max_bound()
        gripper_min_absolute_pos = gripper_joint.min_bound()
        try:
            val = gripper_joint.move(relative_position * (gripper_max_absolute_pos - gripper_min_absolute_pos) + gripper_min_absolute_pos, True)
            return val
        except:
            return False

if __name__ == '__main__':
    arm_controller = ArmController()

    rospy.spin()
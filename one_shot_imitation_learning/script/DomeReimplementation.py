#!/usr/bin/env python3

import rospy, rospkg, torch
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from gazebo_msgs.srv import GetModelState
from one_shot_imitation_learning.srv import GetPose
import tf.transformations as tf
from network.UNetWithTilingAndFiLM import UNetWithTilingAndFiLM
from network.visual_servoing_networks import SiameseExEy, SiameseEz, SiameseEr
from PIL import Image as PILImage
from torchvision import transforms
import numpy as np

class DomeReimplementation:
    
    TOLERENCE_LINEAR = 0.01
    TOLERENCE_ANGULAR = 0.02
    MAX_STEP_SIZE_LINEAR = 0.1   # Maximum linear movement stepsize
    MAX_STEP_SIZE_ANGULAR = 0.2   # Maximum angular movement stepsize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64, 64)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    def __init__(self, weight_path_seg, weight_path_exey, weight_path_ez, weight_path_er, bottleneck_image_path, target_model_name, device):

        # Segmentation Network
        self.segmentation_network = UNetWithTilingAndFiLM().to(device)
        self.segmentation_network.load_state_dict(torch.load(weight_path_seg, weights_only=True))
        self.segmentation_network.eval()

        # Visual Servoing Network
        self.visual_servoing_network_exey = SiameseExEy().to(device)
        self.visual_servoing_network_exey.load_state_dict(torch.load(weight_path_exey, weights_only=True))
        self.visual_servoing_network_exey.eval()
        self.visual_servoing_network_ez = SiameseEz().to(device)
        self.visual_servoing_network_ez.load_state_dict(torch.load(weight_path_ez, weights_only=True))
        self.visual_servoing_network_ez.eval()
        self.visual_servoing_network_er = SiameseEr().to(device)
        self.visual_servoing_network_er.load_state_dict(torch.load(weight_path_er, weights_only=True))
        self.visual_servoing_network_er.eval()

        # Bottleneck Image
        pil_bottleneck_image = PILImage.open(bottleneck_image_path).convert('RGB')
        self.bottleneck_image = DomeReimplementation.transform(pil_bottleneck_image).unsqueeze(0).to(device)
        output = self.segmentation_network(self.bottleneck_image, self.bottleneck_image)
        self.bottleneck_segmentation = (output > 0.5).to(torch.float32)
        self.bottleneck_segmentation *= 255

        # Save the image for debugging
        # debug_bottleneck_segmentation = self.bottleneck_segmentation.squeeze(0).to(torch.uint8)
        # pil_image = transforms.ToPILImage()(debug_bottleneck_segmentation)
        # pil_image.save("/root/deeprob_project_ws/src/DeepRobProject/one_shot_imitation_learning/script/bottleneck_segmentation.png")
        
        self.device = device
        
        # ROS
        self.target_model_name = target_model_name
        self.reach_goal_count = 0
        self.cv_bridge = CvBridge()
        rospy.wait_for_service('/gazebo/get_model_state')
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        rospy.wait_for_service('get_gripper_pose')
        self.get_gripper_pose = rospy.ServiceProxy('get_gripper_pose', GetPose)
        self.cartesian_pose_pub = rospy.Publisher('pose_displacement', Pose, queue_size=10)
        rospy.sleep(1)
        self.camera_sub = rospy.Subscriber('camera/image_raw', Image, self.callback, queue_size=10)
        rospy.loginfo('Dome Init Completed!')

    def callback(self, image):

        print('image callback')
        cv2_image = self.cv_bridge.imgmsg_to_cv2(image, desired_encoding='rgb8')
        pil_image = PILImage.fromarray(cv2_image)

        live_image = DomeReimplementation.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.segmentation_network(live_image, self.bottleneck_image)
            live_segmentation = (output > 0.5).to(torch.float32)
            live_segmentation *= 255

            # Save the image for debugging
            debug_live_segmentation = live_segmentation.squeeze(0).to(torch.uint8)
            pil_image = transforms.ToPILImage()(debug_live_segmentation)
            pil_image.save("/root/deeprob_project_ws/src/DeepRobProject/one_shot_imitation_learning/script/live_segmentation.png")
        
            # Testing 
            # pil_live_seg = PILImage.open('/root/deeprob_project_ws/src/DeepRobProject/one_shot_imitation_learning/script/live_segmentation.png')
            # live_segmentation = transforms.ToTensor()(pil_live_seg).unsqueeze(0).to(self.device)
            # pil_bot_seg = PILImage.open('/root/deeprob_project_ws/src/DeepRobProject/one_shot_imitation_learning/script/bottleneck_segmentation.png')
            # self.bottleneck_segmentation = transforms.ToTensor()(pil_bot_seg).unsqueeze(0).to(self.device)

            output = self.visual_servoing_network_exey(live_segmentation, self.bottleneck_segmentation)
            ex = output[0][0].item()
            ey = output[0][1].item()
            output = self.visual_servoing_network_ez(live_segmentation, self.bottleneck_segmentation)
            ez = output[0][0].item()
            output = self.visual_servoing_network_er(live_segmentation, self.bottleneck_segmentation)
            er_cos = output[0][0].item()
            er_sin = output[0][1].item()

        print(ex, ey, ez)
        self.move_gripper(ex, ey, ez, er_cos, er_sin)

    def move_gripper(self, ex, ey, ez, er_cos, er_sin):

        # Filter the output
        if abs(ex) > DomeReimplementation.TOLERENCE_LINEAR:
            if ex > DomeReimplementation.MAX_STEP_SIZE_LINEAR:
                ex = DomeReimplementation.MAX_STEP_SIZE_LINEAR
            elif ex < -DomeReimplementation.MAX_STEP_SIZE_LINEAR:
                ex = -DomeReimplementation.MAX_STEP_SIZE_LINEAR
        else:
            ex = 0
        
        if abs(ey) > DomeReimplementation.TOLERENCE_LINEAR:
            if ey > DomeReimplementation.MAX_STEP_SIZE_LINEAR:
                ey = DomeReimplementation.MAX_STEP_SIZE_LINEAR
            elif ey < -DomeReimplementation.MAX_STEP_SIZE_LINEAR:
                ey = -DomeReimplementation.MAX_STEP_SIZE_LINEAR
        else:
            ey = 0

        if abs(ez) > DomeReimplementation.TOLERENCE_LINEAR:
            if ez > DomeReimplementation.MAX_STEP_SIZE_LINEAR:
                ez = DomeReimplementation.MAX_STEP_SIZE_LINEAR
            elif ez < -DomeReimplementation.MAX_STEP_SIZE_LINEAR:
                ez = -DomeReimplementation.MAX_STEP_SIZE_LINEAR
        else:
            ez = 0

        er = np.arctan2(er_sin, er_cos)
        print(er)
        if abs(er) > DomeReimplementation.TOLERENCE_ANGULAR:
            if er > DomeReimplementation.MAX_STEP_SIZE_ANGULAR:
                er = DomeReimplementation.MAX_STEP_SIZE_ANGULAR
            elif er < -DomeReimplementation.MAX_STEP_SIZE_ANGULAR:
                er = -DomeReimplementation.MAX_STEP_SIZE_ANGULAR
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
            self.reach_goal_count += 1
        else:
            self.reach_goal_count = 0
            print(f'ex: {ex}, ey: {ey}, ez: {ez}, er: {er}')
            self.cartesian_pose_pub.publish(displacement)
        
        if self.reach_goal_count >= 3:
            response = self.get_model_state(self.target_model_name, 'world')
            model_pose = response.pose

            response = self.get_gripper_pose()
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
            
            rospy.signal_shutdown('Reached Goal Pose')

if __name__ == '__main__':

    # weight_path_seg = rospkg.RosPack().get_path('one_shot_imitation_learning') + '/script/network/weights/' + 'seg.pth'
    weight_path_seg = rospkg.RosPack().get_path('one_shot_imitation_learning') + '/script/network/weights/' + 'seg_one-shot.pth'
    weight_path_exey = rospkg.RosPack().get_path('one_shot_imitation_learning') + '/script/network/weights/' + 'exey.pth'
    weight_path_ez = rospkg.RosPack().get_path('one_shot_imitation_learning') + '/script/network/weights/' + 'ez.pth'
    weight_path_er = rospkg.RosPack().get_path('one_shot_imitation_learning') + '/script/network/weights/' + 'er.pth'
    bottleneck_image_path = rospkg.RosPack().get_path('one_shot_imitation_learning') + '/script/network/' + 'bottleneck.png'

    # ROS init
    rospy.init_node('dome_reimplementation')
    imitation_netwrok = DomeReimplementation(weight_path_seg, 
                                             weight_path_exey, 
                                             weight_path_ez, 
                                             weight_path_er, 
                                             bottleneck_image_path, target_model_name='Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White', device='cuda')

    rospy.spin()
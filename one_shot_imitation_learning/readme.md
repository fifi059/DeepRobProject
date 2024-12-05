# Collect Dataset

To collect dataset, first you need to setup the environment using the `table.world` file in the `worlds` folder.
Then run the following code:

1. Run `roslaunch one_shot_imitation_learning simulation.launch` and wait until you see the **message: "Arm Controller init Complete!"**, indicating that the robot has reached the pregrasp position. 
- (The first run may take longer to start up Gazebo because it needs to download model files to your local computer.)
2. Run `rosrun one_shot_imitation_learning generate_dataset.py __ns:=gen3`.
- Remember to add **`__ns:=gen3`** at the end of the rosrun command!
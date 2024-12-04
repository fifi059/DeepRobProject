# Collect Dataset

To collect dataset, first you need to setup the environment using the `table.world` file in the `worlds` folder.
Then run the following code:

1. run `roslaunch one_shot_imitation_learning simulation.launch` and wait until robot **reach home position**. (Running the first time will take longer to start up Gazebo, because Gazebo need to download model files to your local computer.)
2. run `rosrun one_shot_imitation_learning ArmController.py __ns:=gen3` and wait until **robot arm facing downward**.
3. run `rosrun one_shot_imitation_learning generate_dataset.py __ns:=gen3`.
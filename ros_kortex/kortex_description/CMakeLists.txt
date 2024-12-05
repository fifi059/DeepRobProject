cmake_minimum_required(VERSION 3.0.2)
project(kortex_description)

# Find catkin and any catkin packages
find_package(catkin REQUIRED)

# Find Gazebo
find_package(gazebo REQUIRED)
find_package(gazebo_ros REQUIRED)

# Include directories
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${GAZEBO_INCLUDE_DIRS}  # Include Gazebo headers
)

# Declare a catkin package
catkin_package()

# Install files
foreach(dir arms grippers robots launch)
  install(DIRECTORY ${dir}/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
endforeach(dir)

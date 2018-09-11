cmake_minimum_required(VERSION 3.5)
project(dbow2)

# enable C++11 standard (current directory scope)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON) # prevent fallback


# git clone the specific dbow2 library from origin
include(ExternalProject)
ExternalProject_Add(dbow2_external_git
	PREFIX ${CMAKE_CURRENT_BINARY_DIR}/dependencies
	GIT_REPOSITORY http://github.com/dorian3d/DBoW2
	GIT_TAG v1.1-nonfree
	INSTALL_DIR ${CATKIN_DEVEL_PREFIX}
	CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> # CATKIN_DEVEL_PREFIX will make .cmake file appear in private devel folder
    # DEPENDS
	)

# This should come before any other packages
find_package(catkin REQUIRED)
catkin_package(
	# INCLUDE_DIRS ${CATKIN_DEVEL_PREFIX}/include
	# LIBRARIES DBoW2 DLib# this is needed to include library names into ${PROJECT_NAME}Config.cmake
	# CATKIN_DEPENDS roscpp std_msgs
	# DEPENDS system_lib
	)


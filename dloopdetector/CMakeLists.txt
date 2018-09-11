cmake_minimum_required(VERSION 3.5)
project(dloopdetector)

# enable C++11 standard (current directory scope)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON) # prevent fallback


### configuration for catkin packages

find_package(catkin REQUIRED COMPONENTS dbow2)

# set(DEPENDENCY_DIR ${CMAKE_BINARY_DIR}/dependencies)
# set(DEPENDENCY_INSTALL_DIR ${DEPENDENCY_DIR}/install)
find_package(DBoW2 REQUIRED)
find_package(DLib REQUIRED)
find_package(OpenCV REQUIRED)

catkin_package(
	# INCLUDE_DIRS
	# LIBRARIES rs2_ros
	# CATKIN_DEPENDS dbow2
	DEPENDS DBoW2 DLib OpenCV# this is just for checking purposes
    )
    

### Package specifics 

option(BUILD_DemoBRIEF  "Build demo application with BRIEF features" ON)

# message (WARNING "CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}")
# message (WARNING "DBoW2_LIBS=${DBoW2_LIBS}")
# message (WARNING "DLib_INCLUDE_DIRS=${DLib_INCLUDE_DIRS}")
# message (WARNING "dbow2_LIBRARIES=${dbow2_LIBRARIES}")
# message (WARNING "catkin_LIBRARIES=${catkin_LIBRARIES}")
# message (WARNING "DBoW2_INCLUDE_DIRS=${DBoW2_INCLUDE_DIRS}")
# message (WARNING "catkin_INCLUDE_DIRS=${catkin_INCLUDE_DIRS}")

if(BUILD_DemoBRIEF)
  add_executable(demo_brief demo/demo_brief.cpp)
  target_include_directories(demo_brief PRIVATE include/DLoopDetector ${DBoW2_INCLUDE_DIRS} ${DLib_INCLUDE_DIRS})
  target_link_libraries(demo_brief PRIVATE ${OpenCV_LIBS} ${DBoW2_LIBS} ${DLib_LIBS} )
endif(BUILD_DemoBRIEF)
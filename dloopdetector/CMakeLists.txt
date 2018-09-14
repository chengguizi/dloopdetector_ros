cmake_minimum_required(VERSION 3.5)
project(dloopdetector)

# enable C++11 standard (current directory scope)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON) # prevent fallback


### configuration for catkin packages

find_package(catkin REQUIRED COMPONENTS 
  roscpp 
  sensor_msgs 
  message_filters 
  image_transport 
  cv_bridge 

  dbow2
)

# set(DEPENDENCY_DIR ${CMAKE_BINARY_DIR}/dependencies)
# set(DEPENDENCY_INSTALL_DIR ${DEPENDENCY_DIR}/install)
find_package(DBoW2 REQUIRED)
find_package(DLib REQUIRED)
find_package(OpenCV REQUIRED)

catkin_package(
	# INCLUDE_DIRS
	# LIBRARIES
	# CATKIN_DEPENDS
	DEPENDS DBoW2 DLib OpenCV# this is just for checking purposes
    )
    

### Package specifics 

option(BUILD_DemoBRIEF  "Build demo application with BRIEF features" ON)

if(BUILD_DemoBRIEF)
  add_executable(demo_brief_stereo demo/demo_brief_stereo.cpp)
  target_include_directories(demo_brief_stereo PRIVATE 
    include/DLoopDetector 
    ${DBoW2_INCLUDE_DIRS} 
    ${DLib_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
  )

  target_link_libraries(demo_brief_stereo PRIVATE 
    ${OpenCV_LIBS} 
    ${DBoW2_LIBS} 
    ${DLib_LIBS} 
    ${catkin_LIBRARIES}
  )
endif(BUILD_DemoBRIEF)
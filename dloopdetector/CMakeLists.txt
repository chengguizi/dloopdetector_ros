cmake_minimum_required(VERSION 3.5)
project(dloopdetector)

# enable C++11 standard (current directory scope)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON) # prevent fallback

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

# set(CMAKE_CXX_FLAGS "-Wall -Wextra")
# set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# may not need this
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

# message("CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}")
# message("CMAKE_CXX_FLAGS is ${CMAKE_CXX_FLAGS}")

### configuration for catkin packages

find_package(catkin REQUIRED COMPONENTS 
  roscpp 
  sensor_msgs 
  message_filters 
  image_transport 
  cv_bridge
  image_geometry

  dbow2
  dlib_ros
  libviso2_eigen
)

# set(DEPENDENCY_DIR ${CMAKE_BINARY_DIR}/dependencies)
# set(DEPENDENCY_INSTALL_DIR ${DEPENDENCY_DIR}/install)
find_package(DBoW2 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

catkin_package(
	# INCLUDE_DIRS
	# LIBRARIES
	# CATKIN_DEPENDS
	# DEPENDS DBoW2 OpenCV EIGEN3 #  DLib this is just for checking purposes
    )
    

### Package specifics 

option(BUILD_DemoBRIEF  "Build demo application with BRIEF features" ON)

if(BUILD_DemoBRIEF)
  add_executable(demo_brief_stereo demo/demo_brief_stereo.cpp)
  target_include_directories(demo_brief_stereo PRIVATE 
    include 
    ${EIGEN3_INCLUDE_DIRS}
    ${DBoW2_INCLUDE_DIRS} 
    ${catkin_INCLUDE_DIRS}
    ${CATKIN_DEVEL_PREFIX}/../../include # to fix bug that include folder doesnt exist at configure time
  )

  target_link_libraries(demo_brief_stereo PRIVATE 
    ${OpenCV_LIBS} 
    ${DBoW2_LIBS} 
    ${catkin_LIBRARIES}
  )
endif(BUILD_DemoBRIEF)
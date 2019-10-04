cmake_minimum_required (VERSION 2.8)
project (calib_evaluation)

#set(CMAKE_BUILD_TYPE Debug)
set(BUILD_TYPE Release)

set(EXECUTABLE_OUTPUT_PATH bin)
set(LIBRARY_OUTPUT_PATH lib)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_FLAGS "-std=c++11")

# required libraries
find_package(OpenCV REQUIRED)

#configure a header file to pass some of the CMAKE settings
#to the source code
include_directories(
    ${PROJECT_SOURCE_DIR} 
    ${OpenCV_INCLUDE_DIRS}
)

# Sources files
set(calib_evaluation_SOURCE_FILES
  ${PROJECT_SOURCE_DIR}/src/evalu_repro_error.cpp
)


# compile main library.
#add the executable
add_executable(calib_evaluation ${calib_evaluation_SOURCE_FILES})
target_link_libraries( calib_evaluation ${OpenCV_LIBS})

cmake_minimum_required(VERSION 3.3)

project(aruco_marker)

set( CMAKE_CXX_FLAGS "-std=c++11" )

set( CMAKE_MODULE_PATH ${CMAKE_INSTALL_PREFIX}/lib/cmake/ )

find_library( CPPREST_LIBS cpprest)

find_package( Boost REQUIRED COMPONENTS random system thread filesystem chrono atomic date_time regex )

find_package( OpenSSL REQUIRED )

find_package( OpenCV REQUIRED )

find_package( aruco REQUIRED )

add_executable(aruco_marker main.cpp)

add_executable(aruco_marker_map aruco_marker_map.cpp)

add_executable(calibrate_camera calibrate_camera.cpp)

include_directories( ${OpenCV_INCLUDE_DIRS} )

target_link_libraries( aruco_marker ${OpenCV_LIBS} ${aruco_LIBS})

target_link_libraries( aruco_marker_map ${OpenCV_LIBS} ${aruco_LIBS}
	${CPPREST_LIBS}
	Boost::boost
	Boost::random
	Boost::system
	Boost::thread
	Boost::filesystem
	Boost::chrono
	Boost::atomic
	Boost::date_time
	Boost::regex
	OpenSSL::SSL)

target_link_libraries( calibrate_camera ${OpenCV_LIBS} )
#Specify the version being used aswell as the language
cmake_minimum_required(VERSION 3.19)

#Name your project here
project(tensor LANGUAGES CXX CUDA)


#set the module directory
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")
set(CMAKE_CXX_STANDARD 20)

find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED core glfw-binding opengl3-binding)
find_package(GLEW REQUIRED)
find_package(TIRA REQUIRED)
find_package(Eigen3 REQUIRED)
set(Boost_NO_WARN_NEW_VERSIONS 1)
find_package(Boost REQUIRED COMPONENTS program_options)


#build the executable in the binary directory on MS Visual Studio
if ( MSVC )
	SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIRECTORY}")
	SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIRECTORY}")
	SET( LIBRARY_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIRECTORY}")
	SET( LIBRARY_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIRECTORY}")
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
	add_definitions(-D_SCL_SECURE_NO_WARNINGS)
	add_definitions(-D_USE_MATH_DEFINES)
endif ( MSVC )

configure_file(Roboto-Medium.ttf 
				Roboto-Medium.ttf COPYONLY)

# copy test data
configure_file(data/spread2_hurl5.bmp 
				spread2_hurl5.bmp COPYONLY)
configure_file(data/st2d_spread2_hurl5.npy 
				st2d_spread2_hurl5.npy COPYONLY)

# copy python scripts
configure_file(structen.py 
				structen.py COPYONLY)
configure_file(tensorvote.py 
				tensorvote.py COPYONLY)
				
#set the include directories
include_directories(
			${CMAKE_CURRENT_BINARY_DIR}
			${CMAKE_CURRENT_SOURCE_DIR}
			${GLFW_INCLUDE_DIRS}
			${TIRA_INCLUDE_DIRS}
)

# tensorview executable
add_executable(tensorview2
				tensorview2.cpp
				ImGuiFileDialog/ImGuiFileDialog.cpp
)

target_link_libraries(tensorview2
				PRIVATE glfw
				PRIVATE glm::glm
				PRIVATE GLEW::GLEW
				${OPENGL_LIBRARIES}
				${CMAKE_DL_LIBS}
				PRIVATE imgui::imgui
)

#create an executable
add_executable(tensorview3
				tensorview3.cpp
				gui.cpp
				gui.h
				ImGuiFileBrowser/ImGuiFileBrowser.cpp
				ImGuiFileBrowser/ImGuiFileBrowser.h
				ImGuiFileBrowser/dirent.h
)

target_link_libraries(tensorview3
				PRIVATE glm::glm
				PRIVATE glfw
				PRIVATE GLEW::GLEW
				${OPENGL_LIBRARIES}
				${CMAKE_DL_LIBS}
				Boost::program_options
#				Eigen3::Eigen
				PRIVATE imgui::imgui
)

#create an executable
add_executable(structen
				structen.cpp
)

target_link_libraries(structen
				Boost::program_options
				Eigen3::Eigen
)

# tensorvote executable
add_executable(tensorvote
				tensorvote.cpp
				tensorvote.h
				tensorvote.cuh	
				tensorvote.cu
)

target_link_libraries(tensorvote
				PRIVATE glm::glm
				Boost::program_options
)


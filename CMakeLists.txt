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
endif ( MSVC )

configure_file(Roboto-Medium.ttf 
				Roboto-Medium.ttf COPYONLY)

configure_file(data/lena.npy 
				lena.npy COPYONLY)

configure_file(shaders/colormap.shader 
				colormap.shader COPYONLY)

configure_file(data/oval.npy 
				oval.npy COPYONLY)

configure_file(data/0_1.npy 
				0_1.npy COPYONLY)

configure_file(data/1_1.npy 
				1_1.npy COPYONLY)
configure_file(data/spread_0.bmp 
				spread_0.bmp COPYONLY)

configure_file(data/damaged_shape.npy 
				damaged_shape.npy COPYONLY)

configure_file(shaders/glyph2d.shader 
				glyph2d.shader COPYONLY)
configure_file(shaders/source.shader 
				source.shader COPYONLY)
configure_file(shaders/volume.shader 
				volume.shader COPYONLY)

# copy python scripts
configure_file(structure2d.py 
				structure2d.py COPYONLY)
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

# tensorvote executable
add_executable(tensorvote
				tensorvote.cuh	
				tensorvote.cu
)

target_link_libraries(tensorvote
				PRIVATE glm::glm
)

# tensorview 3 components
configure_file(data/oval3d.npy 
			   oval3d.npy COPYONLY)


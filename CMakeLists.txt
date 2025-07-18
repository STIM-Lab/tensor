#Specify the version being used as well as the language
cmake_minimum_required(VERSION 3.29)
cmake_policy(VERSION 3.29)

if(POLICY CMP0167)
   cmake_policy(SET CMP0167 OLD)
endif()

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
find_package(Boost REQUIRED COMPONENTS program_options math)
#find_package(Boost CONFIG)
find_package(PNG REQUIRED)
find_package (JPEG REQUIRED)


#build the executable in the binary directory on MS Visual Studio
if ( MSVC )
	SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIRECTORY}")
	SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIRECTORY}")
	SET( LIBRARY_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIRECTORY}")
	SET( LIBRARY_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIRECTORY}")
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
	add_definitions(-D_SCL_SECURE_NO_WARNINGS)
	add_definitions(-D_USE_MATH_DEFINES)
else()
	find_package(X11 REQUIRED)
	set(CMAKE_CXX_FLAGS "-Wall -Wextra")
	set(CMAKE_CXX_FLAGS_RELEASE "-O3")
	set(CMAKE_CXX_FLAGS_DEBUG "-g")
endif ( MSVC )


configure_file(Roboto-Medium.ttf Roboto-Medium.ttf COPYONLY)

# copy python scripts
configure_file(image2tensor.py 
				image2tensor.py COPYONLY)
configure_file(tensorvote.py 
				tensorvote.py COPYONLY)
configure_file(tensorvote3d.py 
				tensorvote3d.py COPYONLY)
configure_file(gendata.py 
				gendata.py COPYONLY)
configure_file(physarum.bmp
				physarum.bmp COPYONLY)
configure_file(demo2.png
		demo2.png COPYONLY)
configure_file(demo2noise.png
		demo2noise.png COPYONLY)
file(COPY sbfsem DESTINATION .)

#set the include directories
include_directories(
			${CMAKE_CURRENT_BINARY_DIR}
			${CMAKE_CURRENT_SOURCE_DIR}
			${GLFW_INCLUDE_DIRS}
			${TIRA_INCLUDE_DIRS}
			${X11_INCLUDE_DIR}
			JPEG::JPEG
)

# tensorview executable
add_executable(tvote2
				tvote2/tvote2.cpp
				tvote2/tvote2.h
				ImGuiFileDialog/ImGuiFileDialog.cpp
				tvote2/tvote2_render.cpp
				tvote2/tvote2_imgui.cpp
				tvote2/tvote2_processing.cpp
				tvote2/tvote2_io.cpp
				tvote2/tvote2_callbacks.cpp
				tvote2/tvote2_hsa.cu
)

target_link_libraries(tvote2
				PRIVATE glfw
				PRIVATE glm::glm
				PRIVATE GLEW::GLEW
				${OPENGL_LIBRARIES}
				${CMAKE_DL_LIBS}
				Boost::program_options
				PRIVATE imgui::imgui
				PNG::PNG
				JPEG::JPEG
)

#create an executable
add_executable(tvote3
				tvote3/tvote3.cpp
				tvote3/tvote3.h
				tvote3/tvote3_render.cpp
				tvote3/tvote3_imgui.cpp
				tvote3/tvote3_processing.cpp
				tvote3/tvote3_io.cpp
				tvote3/tvote3_callbacks.cpp
				tvote3/tvote3_hsa.cu
				#cuda_calls.cu
				#tensorvote.cu
				ImGuiFileDialog/ImGuiFileDialog.cpp
)

target_link_libraries(tvote3
				PRIVATE glm::glm
				PRIVATE glfw
				PRIVATE GLEW::GLEW
				${OPENGL_LIBRARIES}
				${CMAKE_DL_LIBS}
				Boost::program_options
				PRIVATE imgui::imgui
)

#create an executable
add_executable(image2tensor
				image2tensor.cpp
				cuda_calls.cu
)

target_link_libraries(image2tensor
				Boost::program_options
				${X11_LIBRARIES}
				PNG::PNG
				JPEG::JPEG

)

# tensorvote executable
add_executable(tensorvote
				tensorvote.cpp
				tensorvote.cuh	
				tensorvote.cu
				cuda_calls.cu
)

target_link_libraries(tensorvote
				PRIVATE glm::glm
				Boost::program_options
				Boost::math
)

target_compile_options(tvote3 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
		--generate-line-info
		>)


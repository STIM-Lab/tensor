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
find_package(imgui CONFIG REQUIRED core glfw-binding opengl3-binding)
find_package(GLEW REQUIRED)
find_package(TIRA REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options math)
find_package(glm CONFIG REQUIRED)
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

# suppress warnings regarding ImGuiFileDialog, which we didn't write
set_source_files_properties(
		ImGuiFileDialog/ImGuiFileDialog.cpp
		PROPERTIES
		COMPILE_FLAGS "-Wno-deprecated-enum-enum-conversion"
)

#configure_file(Roboto-Medium.ttf Roboto-Medium.ttf COPYONLY)

# copy python scripts
configure_file(python/image2tensor.py
				image2tensor.py COPYONLY)
configure_file(python/tensorvote.py
				tensorvote.py COPYONLY)
configure_file(python/tensorvote3d.py
				tensorvote3d.py COPYONLY)
configure_file(data/gendata.py
				gendata.py COPYONLY)
configure_file(data/physarum.bmp
				physarum.bmp COPYONLY)
configure_file(data/demo2.png
		demo2.png COPYONLY)
configure_file(data/demo2noise.png
		demo2noise.png COPYONLY)
configure_file(data/impulse_blur_13.npy
		impulse_blur_13.npy COPYONLY)
configure_file(data/bar_tensor_field.npy
		bar_tensor_field.npy COPYONLY)
file(COPY data/sbfsem DESTINATION .)

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
				source/tvote2/tvote2.cpp
				source/tvote2/tvote2.h
				source/ImGuiFileDialog/ImGuiFileDialog.cpp
				source/ImGuiFileDialog/ImGuiFileDialog.h
				source/tvote2/tvote2_render.cpp
				source/tvote2/tvote2_imgui.cpp
				source/tvote2/tvote2_processing.cpp
				source/tvote2/tvote2_io.cpp
				source/tvote2/tvote2_callbacks.cpp
				source/tvote2/tvote2_hsa.cu
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
		source/tvote3/tvote3.cpp
		source/tvote3/tvote3.h
		source/tvote3/tvote3_render.cpp
		source/tvote3/tvote3_imgui.cpp
		source/tvote3/tvote3_processing.cpp
		source/tvote3/tvote3_io.cpp
		source/tvote3/tvote3_callbacks.cpp
		source/tvote3/tvote3_hsa.cu
		source/ImGuiFileDialog/ImGuiFileDialog.cpp
		source/ImGuiFileDialog/ImGuiFileDialog.h
		source/include/glyphs.h
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
add_executable(glyphs3
		source/glyphs3/glyphs3.cpp
		source/include/glyphs.h
)

target_link_libraries(glyphs3
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
				source/image2tensor/image2tensor.cpp
				source/image2tensor/cuda_calls.cu
)

target_link_libraries(image2tensor
				Boost::program_options
				${X11_LIBRARIES}
				PNG::PNG
				JPEG::JPEG

)


target_compile_options(tvote3 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
		--generate-line-info
		>)


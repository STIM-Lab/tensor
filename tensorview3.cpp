#include "gui.h"
#include <numbers>
#include <complex>
#include <GL/glew.h>
#include "volumetric.h"
#include <GLFW/glfw3.h>
#include "tira/volume.h"
#include "tira/graphics_gl.h"
#include <glm/gtc/quaternion.hpp>
#include <boost/program_options.hpp>

#define PI 3.14159265358979323846

GLFWwindow* window;                                     // pointer to the GLFW window that will be created (used in GLFW calls to request properties)
const char* glsl_version = "#version 130";              // specify the version of GLSL
tira::camera camera;

extern bool perspective;
float move[2] = { 0.0f, 0.0f };							// UP and RIGHT, respectively
bool ctrl = false;
bool dragging = false;
double xprev, yprev;
size_t axes[] = { 0, 0, 0 };
int scroll_value = 0;

tira::volumetric<glm::mat3> T;							// 3D tensor field (3x3 voxels)
tira::volume<unsigned char> V;							// 3D raw volume
int gui_VolumeSize[] = { 0, 0, 0 };
float gui_PixelSize[] = { 1.0f, 1.0f, 1.0f };

// input variables for arguments
std::string in_filename;
std::string in_image;
float in_gamma;
int in_cmap;
int step = 4;

tira::glMaterial* GLYPH_MATERIAL;
tira::glMaterial* VOLUME_MATERIAL;

const std::string glyph_shader_string =
#include "shaders/glyph3d.shader"
;
const std::string volume_shader_string =
#include "shaders/volume.shader"
;

void glfw_error_callback(int error, const char* description)
{
	fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS && window_focused)
		dragging = true;
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
		dragging = false;
	glfwGetCursorPos(window, &xprev, &yprev);
}

static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
	double ANGLE_SCALE = 0.005;
	if (dragging) {
		double dx = xpos - xprev;
		double dy = ypos - yprev;
		camera.orbit(-dx * ANGLE_SCALE, dy * ANGLE_SCALE);
		xprev = xpos;
		yprev = ypos;
	}
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
	if (TENSOR_LOADED) {
		scroll_value += yoffset;
		if (scroll_value < 0) scroll_value = 0;
		if (scroll_value >= T.Z()) scroll_value = T.Z() - 1;
		std::cout << scroll_value << std::endl;
	}
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
		perspective = !perspective;
	}
	if (key == GLFW_KEY_LEFT_CONTROL && action == GLFW_PRESS) {
		ctrl = true;
	}
	if (key == GLFW_KEY_LEFT_CONTROL && action == GLFW_RELEASE) {
		ctrl = false;
	}
	if (key == GLFW_KEY_KP_SUBTRACT && action == GLFW_PRESS && ctrl) {
		zoom -= 0.1;
		if (zoom < 0.1) zoom = 0.1;
	}
	if (key == GLFW_KEY_KP_ADD && action == GLFW_PRESS && ctrl) {
		zoom += 0.1;
		if (zoom > 2) zoom = 2;
	}
	if (key == GLFW_KEY_UP && action == GLFW_PRESS) {
		move[0] += 5;
		if (move[0] > 100) move[0] = 100;
	}
	if (key == GLFW_KEY_DOWN && action == GLFW_PRESS) {
		move[0] -= 5;
		if (move[0] < -100) move[0] = -100;
	}
	if (key == GLFW_KEY_RIGHT && action == GLFW_PRESS) {
		move[1] += 5;
		if (move[1] > 100) move[1] = 100;
	}
	if (key == GLFW_KEY_LEFT && action == GLFW_PRESS) {
		move[1] -= 5;
		if (move[1] < -100) move[1] = -100;
	}
}

void resetPlane(float frame) {
	// Reset camera view to initial state
	camera.position(glm::vec3(0, 0, 0));
	camera.lookat(glm::vec3(0, 0, 1), glm::vec3(0, 1, 0));
	camera.fov(60.f);

	move[1] = 0.0f;
	move[0] = 0.0f;
	zoom = 1.0f;
	axes[0] = 0; axes[1] = 0; axes[2] = 0;
	scroll_axis = 2;
	scroll_value = 0;
	anisotropy = 0;
	filter = 0.1f;
	thresh = 0.0f;
	step = 4;
}

GLFWwindow* InitGLFW() {
	GLFWwindow* window;

	// Setup window
	glfwSetErrorCallback(glfw_error_callback);
	if (!glfwInit())
		return NULL;

	// GL 3.0 + GLSL 130
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

	// Create window with graphics context
	window = glfwCreateWindow(1600, 1200, "GLFW+OpenGL3 Hello World Program", NULL, NULL);
	if (window == NULL)
		return NULL;
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1); // Enable vsync

	// input callback functions
	glfwSetMouseButtonCallback(window, mouse_button_callback);
	glfwSetCursorPosCallback(window, cursor_position_callback);
	glfwSetScrollCallback(window, scroll_callback);
	glfwSetKeyCallback(window, key_callback);
	return window;
}

void InitGLEW() {
	GLenum err = glewInit();
	if (GLEW_OK != err) {
		/* Problem: glewInit failed, something is seriously wrong. */
		fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
		exit(1);
	}
}

tira::volume<float> GetDiagValues(tira::volume<glm::mat3> T) {
	tira::volume<float> diagonal_elem(T.X(), T.Y(), T.Z(), 3);
	glm::mat3 tensor;
	for (size_t m = 0; m < T.X(); m++) {
		for (size_t n = 0; n < T.Y(); n++) {
			for (size_t p = 0; p < T.Z(); p++) {
				tensor = T(m, n, p);
				diagonal_elem(m, n, p, 0) = tensor[0][0];
				diagonal_elem(m, n, p, 1) = tensor[1][1];
				diagonal_elem(m, n, p, 2) = tensor[2][2];
			}
		}
	}

	return diagonal_elem;
}

tira::volume<float> GetOffDiagValues(tira::volume<glm::mat3> T) {
	tira::volume<float> triangular_elem(T.X(), T.Y(), T.Z(), 3);
	glm::mat3 tensor;
	for (size_t m = 0; m < T.X(); m++) {
		for (size_t n = 0; n < T.Y(); n++) {
			for (size_t p = 0; p < T.Z(); p++) {
				tensor = T(m, n, p);
				triangular_elem(m, n, p, 0) = tensor[0][1];
				triangular_elem(m, n, p, 1) = tensor[0][2];
				triangular_elem(m, n, p, 2) = tensor[1][2];
			}
		}
	}

	return triangular_elem;
}

// Load a tensor field from a NumPy file
void LoadTensorField3(std::string npy_filename) {
	// Load the tensor field
	T.load_npy<float>(npy_filename);
	// Separate the diagonal and off-diagonal elements to be sent off to GPU as RGB volume texture-maps
	tira::volume<float> diagonal_elem = GetDiagValues(T);
	tira::volume<float> triangular_elem = GetOffDiagValues(T);

	// Copy the tensor field (including diagonal and off-diagonal RGB volumes) to GPU as texture maps
	GLYPH_MATERIAL->SetTexture("Diagonal", diagonal_elem, GL_RGBA32F, GL_LINEAR);
	GLYPH_MATERIAL->SetTexture("Upper_trian", triangular_elem, GL_RGBA32F, GL_LINEAR);

	// save everything for rendering
	TENSOR_LOADED = true;
}

// Load an image from a NumPy file
void LoadVolume3(std::string npy_filename) {
	// Load the image
	V.load_npy(npy_filename);

	VOLUME_MATERIAL->SetTexture("volumeTexture", V, GL_RGB8, GL_NEAREST);

	// save everything for rendering
	VOLUME_LOADED = true;
}

int main(int argc, char** argv) {

	// Declare the supported options.
	boost::program_options::options_description desc("Allowed options");
	desc.add_options()
		("input", boost::program_options::value<std::string>(&in_filename), "input filename for the tensorfields")
		("volume", boost::program_options::value<std::string>(&in_image), "optional image field corresponding to the tensors")
		("gamma, g", boost::program_options::value<float>(&in_gamma)->default_value(3), "glyph gamma (sharpness), 0 = spheroids")
		("cmap,c", boost::program_options::value<int>(&in_cmap)->default_value(0), "colormaped eigenvector (0 = longest, 2 = shortest)")
		("help", "produce help message");
	boost::program_options::variables_map vm;

	boost::program_options::positional_options_description p;
	p.add("input", -1);
	boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
	boost::program_options::notify(vm);

	if (vm.count("help")) {
		std::cout << desc << std::endl;
		return 1;
	}

	if (in_cmap != 0 && in_cmap != 2) {
		std::cout << "ERROR: invalid colormap component. Options (0 = longest, 2 = shortest)." << std::endl;
		return 1;
	}

	// Initialize OpenGL
	window = InitGLFW();                                // create a GLFW window
	InitUI(window, glsl_version);
	InitGLEW();

	// Enable OpenGL environment parameters
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);

	tira::glGeometry glyph = tira::glGeometry::GenerateIcosphere<float>(3, false);			// create a glyph
	tira::glGeometry rectangle = tira::glGeometry::GenerateRectangle<float>();				// create a rectangle for rendering volume
	GLYPH_MATERIAL = new tira::glMaterial(glyph_shader_string);
	VOLUME_MATERIAL = new tira::glMaterial(volume_shader_string);

	// If the tensor field is loaded using command-line argument
	if (vm.count("input")) {
		LoadTensorField3(in_filename);
		OPEN_TENSOR = false;
		std::cout << "Tensor loaded successfully\n" << std::endl;
		std::cout << "Size of volume:\t" << T.X() << " x " << T.Y() << " x " << T.Z() << std::endl;
	}

	// If an image volume is specified, load it as a texture
	if (vm.count("volume")) {
		LoadVolume3(in_image);
		OPEN_VOLUME = false;
		std::cout << "Volume loaded successfully\n" << std::endl;
		//I.load_npy(in_image);
		//VOLUME_MATERIAL.SetTexture("volumeTexture", I, GL_RGB, GL_NEAREST);
	}

	float gamma = in_gamma;
	int component_color = in_cmap;

	// Set light position
	glm::vec4 light0(0.0f, 100.0f, 100.0f, 0.7f);
	glm::vec4 light1(0.0f, -100.0f, 0.0f, 0.5f);
	float ambient = 0.3;


	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();													// Poll and handle events (inputs, window resize, etc.)
		RenderUI();
		int display_w, display_h;                                           // size of the frame buffer (openGL display)
		glfwGetFramebufferSize(window, &display_w, &display_h);

		float aspect = (float)display_w / (float)display_h;
		glm::mat4 Mprojection;
		glm::mat4 Mview;

		// If the load command for tensor field is called from ImGui file dialog
		if (OPEN_TENSOR) {
			LoadTensorField3(TensorFileName);								// Load the tensor field and set the texture-map
			OPEN_TENSOR = false;
			std::cout << "Tensor loaded successfully\n" << std::endl;
			std::cout << T.X() << ", " << T.Y() << ", " << T.Z() << std::endl;
			std::cout << T.dx() << ", " << T.dy() << ", " << T.dz() << std::endl;
			std::cout << T.sx() << ", " << T.sy() << ", " << T.sz() << std::endl;

		}
		// If the load command for volume is called from ImGui file dialog
		if (OPEN_VOLUME) {
			LoadVolume3(VolumeFileName);									// Load the volume and set the texture-map
			OPEN_VOLUME = false;
			std::cout << "Volume loaded successfully\n" << std::endl;
		}

		float frame;
		if (TENSOR_LOADED) {
			gui_VolumeSize[0] = T.sx();
			gui_VolumeSize[1] = T.sy();
			gui_VolumeSize[2] = T.sz();

			T.set_spacing((double)gui_PixelSize[0], (double)gui_PixelSize[1], (double)gui_PixelSize[2]);

			frame = (scroll_axis == 2) ? std::max(T.X(), T.Y()) : ((scroll_axis == 1) ? std::max(T.X(), T.Z()) : std::max(T.Y(), T.Z()));

			// Reset the visualization to initial state if reset button is pushed
			if (RESET) resetPlane(frame);

			if (aspect > 1) {
				if (!perspective)
					Mprojection = glm::ortho(-aspect * frame / (2.0f * zoom) + move[1], aspect * frame / (2.0f * zoom) + move[1], -frame / (2.0f * zoom) + move[0], frame / (2.0f * zoom) + move[0], -2.0f * frame, 2.0f * frame);
				else
					Mprojection = glm::perspective(60.0f * (float)std::numbers::pi / 180.0f, aspect, 0.1f, 4.0f * frame);
			}
			else {
				if (!perspective)
					Mprojection = glm::ortho(-frame / 2.0f, frame / 2.0f, -frame / 2.0f / aspect, frame / 2.0f / aspect, -2.0f * frame, 2.0f * frame);
				else
					Mprojection = glm::perspective(60.0f * (float)std::numbers::pi / 180.0f, aspect, 0.1f, 4.0f * frame);
			}

			Mview = camera.matrix();						// generate a view matrix from the camera
		}

		glViewport(0, 0, display_w, display_h);									// specifies the area of the window where OpenGL can render
		glClearColor(0, 0, 0, 0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glm::mat4 Mtran;

		// --------- Render the volume slice (image) ----------------------
		if (VOLUME_LOADED && RENDER_IMAGE) {
			// Enable alpha blending for transparency and set blending function
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

			float z = scroll_value * T.dz() - T.sz() / 2.0f;
			Mtran = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, z));

			// Scale matrix
			glm::mat4 Mscale = glm::scale(glm::mat4(1.0f), glm::vec3(T.sx(), T.sy(), 1.0f));

			// Scroll_value should get mapped from range [0, V.Z() - 1] to range [0, 1]
			float mappep_scroll_value = (float)scroll_value / (float)(V.Z() - 1);
			VOLUME_MATERIAL->Begin();
			VOLUME_MATERIAL->SetUniformMat4f("MVP", Mprojection * Mview * Mtran * Mscale);
			VOLUME_MATERIAL->SetUniform1f("slider", mappep_scroll_value);
			VOLUME_MATERIAL->SetUniform1f("opacity", opacity);
			rectangle.Draw();
			VOLUME_MATERIAL->End();

			glDisable(GL_BLEND);
		}

		glClear(GL_DEPTH_BUFFER_BIT);

		component_color = (cmap) ? 2 : 0;

		GLYPH_MATERIAL->Begin();
		GLYPH_MATERIAL->SetUniformMat4f("MV", Mprojection * Mview);
		GLYPH_MATERIAL->SetUniform4f("light0", light0);
		GLYPH_MATERIAL->SetUniform4f("light1", light1);
		GLYPH_MATERIAL->SetUniform1f("ambient", ambient);
		GLYPH_MATERIAL->SetUniform1i("ColorComponent", component_color);
		GLYPH_MATERIAL->SetUniform1f("gamma", gamma);
		GLYPH_MATERIAL->SetUniform1i("size", step);
		GLYPH_MATERIAL->SetUniform1f("filter", filter);
		GLYPH_MATERIAL->SetUniform1i("anisotropy", anisotropy);
		GLYPH_MATERIAL->SetUniform1f("thresh", thresh);

		// Rendering the tensor field for the selected axis
		if (TENSOR_LOADED && RENDER_GLYPHS) {

			for (size_t yi = 0; yi < T.Y(); yi += step) {
				for (size_t xi = 0; xi < T.X(); xi += step) {

					// calculate the position of the current glyph
					float y = yi * T.dy() - T.sy() / 2.0f + (T.dy() * step) / 2.0f;
					float x = xi * T.dx() - T.sx() / 2.0f + (T.dx() * step) / 2.0f;
					float z = scroll_value * T.dz() - T.sz() / 2.0f;

					// render the glyph at position (x, y, z)
					Mtran = glm::translate(glm::mat4(1.0f), glm::vec3(x, y, z));
					GLYPH_MATERIAL->SetUniformMat4f("Mtran", Mtran);
					GLYPH_MATERIAL->SetUniform3ui("position", xi, yi, scroll_value);
					glyph.Draw();

				}
			}
		}

		GLYPH_MATERIAL->End();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());     // draw the GUI data from its buffer
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	DestroyUI();
	glfwDestroyWindow(window);                                      // Destroy the GLFW rendering window
	glfwTerminate();                                                // Terminate GLFW

}
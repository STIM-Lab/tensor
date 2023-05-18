#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"


extern bool reset;
extern bool window_focused;


void InitUI(GLFWwindow* window, const char* glsl_version);
void DestroyUI();
void RenderUI();
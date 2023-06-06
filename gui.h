#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"


extern bool reset;
extern int scroll_axis;
extern bool window_focused;
extern bool axis_change;


void InitUI(GLFWwindow* window, const char* glsl_version);
void DestroyUI();
void RenderUI();
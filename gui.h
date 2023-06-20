#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "lib/ImGuiFileDialog/ImGuiFileDialog.h"

extern bool reset;
extern int scroll_axis;
extern bool window_focused;
extern bool axis_change;
extern int anisotropy;
extern float filter;
extern float zoom;
extern int cmap;
extern bool menu_open;
extern bool image_plane;
extern float opacity;
extern float thresh;

void InitUI(GLFWwindow* window, const char* glsl_version);
void DestroyUI();
void RenderUI();
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "lib/ImGuiFileDialog/ImGuiFileDialog.h"

extern bool reset;
extern int scroll_axis;
extern bool window_focused;
extern bool axis_change;
extern int anisotropy;
extern float accuracy;
extern float zoom;
extern bool cmap;
extern bool menu_open;
extern bool image_plane;

void InitUI(GLFWwindow* window, const char* glsl_version);
void DestroyUI();
void RenderUI();
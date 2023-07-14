#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "ImGuiFileBrowser/ImGuiFileBrowser.h"

extern bool RESET;
extern int scroll_axis;
extern bool window_focused;
extern bool axis_change;
extern int anisotropy;
extern float filter;
extern float zoom;
extern int cmap;
extern float opacity;
extern float thresh;

extern bool OPEN_TENSOR;
extern bool RENDER_GLYPHS;
extern bool TENSOR_LOADED;
extern std::string TensorFileName;

extern bool OPEN_VOLUME;
extern bool RENDER_IMAGE;
extern bool VOLUME_LOADED;
extern std::string VolumeFileName;


void InitUI(GLFWwindow* window, const char* glsl_version);
void DestroyUI();
void RenderUI();
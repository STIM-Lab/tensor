#include "tvote2.h"
#include <GLFW/glfw3.h> // Will drag system OpenGL headers

extern TV2_UI UI;
extern tira::image<glm::mat2> Tn;

void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

void mouse_button_callback(GLFWwindow* window, const int button, const int action, const int mods)  {
    (void)mods;                                                         // mods is required for the callback but unused
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);
        UI.raw_mouse_position[0] = static_cast<float>(xpos);
        UI.raw_mouse_position[1] = static_cast<float>(ypos);
    }
}

void mouse_callback(GLFWwindow* window, double xposIn, double yposIn) {
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS &&
        glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
        //int state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
        const double xdist = (UI.raw_mouse_position[0] - xposIn) / display_w * UI.viewport[0] * UI.camera_zoom;
        const double ydist = (UI.raw_mouse_position[1] - yposIn) / display_h * UI.viewport[1] * UI.camera_zoom;
        UI.camera_position[0] += static_cast<float>(xdist);
        UI.camera_position[1] += static_cast<float>(ydist);
        UI.raw_mouse_position[0] = static_cast<float>(xposIn);
        UI.raw_mouse_position[1] = static_cast<float>(yposIn);
    }
    else {
        // avoid accessing unavailable information if the field isn't loaded
        if (UI.field_loaded) {
            const float x_adjustment = (UI.viewport[0] - static_cast<float>(Tn.width())) / 2.0f;
            const float y_adjustment = (UI.viewport[1] - static_cast<float>(Tn.height())) / 2.0f;
            UI.field_mouse_position[0] = static_cast<float>(xposIn) / static_cast<float>(display_w) * UI.viewport[0] - x_adjustment;
            UI.field_mouse_position[1] = static_cast<float>(yposIn) / static_cast<float>(display_h) * UI.viewport[1] - y_adjustment;
        }
    }
}

void scroll_callback(GLFWwindow* window, const double xoffset, const double yoffset) {
    (void)xoffset;                                                  // xoffset is required for the callback but unused
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        UI.camera_zoom -= UI.camera_zoom * (static_cast<float>(yoffset) * 0.25f);
}

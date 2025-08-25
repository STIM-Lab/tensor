#include "tvote3.h"

extern TV3_UI UI;

void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    (void) mods;
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS && UI.window_focused)
        UI.mouse_dragging = true;
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
        UI.mouse_dragging = false;
    glfwGetCursorPos(window, &UI.raw_mouse_position[0], &UI.raw_mouse_position[1]);
}

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    (void) window;
    if (UI.mouse_dragging) {
        double ANGLE_SCALE = 0.005;
        double dx = xpos - UI.raw_mouse_position[0];
        double dy = ypos - UI.raw_mouse_position[1];
        UI.camera.orbit(dx * ANGLE_SCALE, dy * ANGLE_SCALE);
        UI.raw_mouse_position[0] = xpos;
        UI.raw_mouse_position[1] = ypos;
    }
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    (void) window;
    (void) xoffset;
    if (UI.field_loaded && UI.window_focused) {
        UI.camera.zoom(0.2 * yoffset);
    }
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    (void) window;
    (void) scancode;
    (void) mods;
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
        UI.perspective = !UI.perspective;
    }
    if (key == GLFW_KEY_LEFT_CONTROL && action == GLFW_PRESS) {
        UI.ctrl_pressed = true;
    }
    if (key == GLFW_KEY_LEFT_CONTROL && action == GLFW_RELEASE) {
        UI.ctrl_pressed = false;
    }
    if (key == GLFW_KEY_KP_SUBTRACT && action == GLFW_PRESS && UI.ctrl_pressed) {
        UI.camera.zoom(-0.1);
    }
    if (key == GLFW_KEY_KP_ADD && action == GLFW_PRESS && UI.ctrl_pressed) {
        UI.camera.zoom(0.1);
    }
}

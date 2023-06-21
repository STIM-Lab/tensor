#include "gui.h"
#include <iostream>


float ui_scale = 1.5f;                                  // scale value for the UI and UI text
bool reset = false;
bool window_focused = true;
bool axis_change = true;                                // gets true when the axis plane is changes
extern int in_size;                                        // the steps between each glyph along all axis
int scroll_axis = 2;				                    // default axis is Z
int anisotropy = 0;                                     // 0: all tensors               1: linear tensors only
                                                        // 2: planar tensors only       3: spherical tensors only
float filter = 0.1f;
float zoom = 1.0f;
int cmap = 2;
bool menu_open = false;
bool image_plane = false;
float opacity = 1.0f;
float thresh = 0.0f;
float move[];

bool CenteredButton(const char* direc, ImGuiStyle& style) {
    float size = ImGui::CalcTextSize(direc).x + style.FramePadding.x * 2.0f;
    float avail = ImGui::GetContentRegionAvail().x;
    float off = (avail - size) * 0.5f;
    if (off > 0.0f)  ImGui::SetCursorPosX(ImGui::GetCursorPosX() + off);
    style.FrameRounding = 10.f;
    if(direc == "D")
        return ImGui::ArrowButton("D", 3);
    else
        return ImGui::ArrowButton("U", 2);
}

/// <summary>
/// Initialize the GUI
/// </summary>
/// <param name="window">Pointer to the GLFW window that will be used for rendering</param>
/// <param name="glsl_version">Version of GLSL that will be used</param>
void InitUI(GLFWwindow* window, const char* glsl_version) {
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    ImGui::GetStyle().ScaleAllSizes(ui_scale);
    ImGui::GetIO().FontGlobalScale = ui_scale;

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Load Fonts
    io.Fonts->AddFontFromFileTTF("Roboto-Medium.ttf", ui_scale * 16.0f);

}

/// <summary>
/// Destroys the ImGui rendering interface (usually called when the program closes)
/// </summary>
void DestroyUI() {
    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

/// <summary>
/// This function renders the user interface every frame
/// </summary>
void RenderUI() {
    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    

    // Display a Demo window showing what ImGui is capable of
    // See https://pthom.github.io/imgui_manual_online/manual/imgui_manual.html for code details
    //ImGui::ShowDemoWindow();

    // Hello World GUI Window
    {

        float old_size = ImGui::GetFont()->Scale;
        ImGui::GetFont()->Scale *= 0.5;
        ImGui::PushFont(ImGui::GetFont());

        ImGui::Begin("Tensor");

        window_focused = (ImGui::IsWindowHovered()) ? false : true;
        
        ImGuiStyle& style = ImGui::GetStyle();
        style.FrameRounding = 5.f;
        style.GrabRounding = 3.f;
        style.WindowRounding = 7.f;

        // Select the number of tensors along X axis
        ImGui::Text("Number of tensors:");      ImGui::SameLine();  ImGui::Text("%d", in_size);
        if(ImGui::Button("-50", ImVec2(75, 25))) {
            in_size -= 50;
            if (in_size < 50)   in_size = 50;
        }
        ImGui::SameLine();
        if (ImGui::Button("+50", ImVec2(75, 25))) {
            in_size += 50;
            if (in_size > 150) in_size = 150;
        }

        // Select which plane to render (view)
        ImGui::Separator();
        ImGui::Text("Plane");
        if (ImGui::RadioButton("xi ", &scroll_axis, 0)) axis_change = true;
        ImGui::SameLine();
        if (ImGui::RadioButton("yi ", &scroll_axis, 1)) axis_change = true;
        ImGui::SameLine();
        if (ImGui::RadioButton("zi ", &scroll_axis, 2)) axis_change = true;

        ImGui::Separator();

        // Filter anisotropy using a threshold filter option
        ImGui::Text("Select Anisotropy");
        ImGui::Columns(2);
        ImGui::RadioButton("ALL", &anisotropy, 0);
        ImGui::RadioButton("Linear", &anisotropy, 1);       ImGui::NextColumn();
        ImGui::RadioButton("Planar", &anisotropy, 2);       
        ImGui::RadioButton("Spherical", &anisotropy, 3);    
        ImGui::Columns(1);
        ImGui::Spacing();
        ImGui::SliderFloat("Filter", &filter, 0.1f, 1.0f);
        ImGui::Separator();

        // Show colormapped image based on shortest/longest eigenvector
        ImGui::Columns(2);
        ImGui::Text("Colormap:");       ImGui::SameLine();
        ImGui::RadioButton("Shortest V", &cmap, 2);    ImGui::NextColumn();
        ImGui::RadioButton("Longest V", &cmap, 0);
        ImGui::Columns(1);
        ImGui::Spacing();
        // Adjust a threshold for eigenvalues corresponding to each tensor
        ImGui::DragFloat("Threshold", &thresh, 0.005f, 1.0f, 2.1f, "%.3f");
        ImGui::Separator();

        // Render the plane with texture-mapped image
        ImGui::Checkbox("Image Plane", &image_plane);
        ImGui::Spacing();
        // Adjust the image transparency
        ImGui::SliderFloat("Opacity", &opacity, 0.1f, 1.0f);
        ImGui::Separator();

        // Zooming in and out option
        ImGui::InputFloat("Zoom", &zoom, 0.1f, 2.0f);
        zoom = (zoom < 1.0f) ? 1.0f : ((zoom > 5) ? 5 : zoom);
        ImGui::SameLine();
        if(ImGui::Button("O", ImVec2(25, 25))) zoom = 1.0f;             // reset zoom

        ImGui::Separator();

        // Reset button
        reset = ImGui::Button("Reset", ImVec2(70, 35));

        ImGui::Separator();

        // Arrow buttons
        style.FrameRounding = 10.f;
        if (CenteredButton("U", style)) {
            move[0] += 5;
            if (move[0] > 100) move[0] = 100;
        }
        // Make sure the buttons are centered
        float size = ImGui::CalcTextSize("\t\t\t\t").x + style.FramePadding.x * 2.f;
        float avail = ImGui::GetContentRegionAvail().x;
        float off = (avail - size) * 0.5f;
        if (off > 0.0f)  ImGui::SetCursorPosX(ImGui::GetCursorPosX() + off);
        if (ImGui::ArrowButton("L", 0)) {
            move[1] -= 5;
            if (move[1] < -100) move[1] = -100;
        }
        style.ItemSpacing = ImVec2(23.0f, 5.0f); ImGui::SameLine();
        if (ImGui::ArrowButton("R", 1)) {
            move[1] += 5;
            if (move[1] > 100) move[1] = 100;
        }
        if (CenteredButton("D", style)) {
            move[0] -= 5;
            if (move[0] < -100) move[0] = -100;
        }
        
        style.FrameRounding = 5.f;
        style.ItemSpacing = ImVec2(8.f, 8.0f);
        
        
        ImGui::GetFont()->Scale = old_size;
        ImGui::PopFont();
        ImGui::End();
    }



    //ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);  // Render a separate window showing the FPS

    ImGui::Render();                                                            // Render all windows
}
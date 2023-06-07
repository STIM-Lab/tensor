#include "gui.h"
#include <iostream>


float ui_scale = 1.5f;                                  // scale value for the UI and UI text
bool reset = false;
bool window_focused = true;
bool axis_change = true;                                // gets true when the axis plane is changes
extern int step;                                        // the steps between each glyph along all axis
int scroll_axis = 2;				                    // default axis is Z
int anisotropy = 0;                                     // 0: all tensors               1: linear tensors only
                                                        // 2: planar tensors only       3: spherical tensors only
float accuracy = 0.1f;
float zoom = 1.0f;
bool cmap = true;
bool menu_open = false;

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
        

        //opnes ImGui File Dialog
        if (ImGui::Button("Open File Dialog"))
        {
            ImGuiFileDialog::Instance()->OpenDialog("ChooseFileDlgKey", "Choose File", ".npy,.cpp,.h,.hpp,.pdf,.bmp", ".");
            menu_open = true;
        }

        if (ImGuiFileDialog::Instance()->Display("ChooseFileDlgKey"))
        {
            if (ImGuiFileDialog::Instance()->IsOk())
            {
                std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();
                std::string filePath = ImGuiFileDialog::Instance()->GetCurrentPath();
            }
            ImGuiFileDialog::Instance()->Close();
        }

        // Adjusting the size of the volume along each axis
        ImGui::SliderInt("Size", &step, 1, 5);
        ImGui::Separator();

        ImGui::Text("Plane");
        if (ImGui::RadioButton("xi ", &scroll_axis, 0)) axis_change = true;
        ImGui::SameLine();
        if (ImGui::RadioButton("yi ", &scroll_axis, 1)) axis_change = true;
        ImGui::SameLine();
        if (ImGui::RadioButton("zi ", &scroll_axis, 2)) axis_change = true;

        ImGui::Separator();

        ImGui::Text("Select Anisotropy");
        ImGui::Columns(2);
        ImGui::RadioButton("ALL", &anisotropy, 0);
        ImGui::RadioButton("Linear", &anisotropy, 1);       ImGui::NextColumn();
        ImGui::RadioButton("Planar", &anisotropy, 2);       
        ImGui::RadioButton("Spherical", &anisotropy, 3);    

        ImGui::Columns(1);
        ImGui::Spacing();
        ImGui::SliderFloat("Accuracy", &accuracy, 0.1f, 1.0f);
        ImGui::Separator();

        ImGui::Checkbox("Colormaped eigenvector", &cmap);
        ImGui::Separator();

        ImGui::InputFloat("Zoom", &zoom, 0.1f, 2.0f);
        if (zoom < 1.0f) zoom = 1.0f;
        if (zoom > 3) zoom = 3;
        ImGui::Separator();

        reset = ImGui::Button("Reset", ImVec2(70, 35));

        ImGui::GetFont()->Scale = old_size;
        ImGui::PopFont();
        ImGui::End();
    }



    //ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);  // Render a separate window showing the FPS

    ImGui::Render();                                                            // Render all windows
}
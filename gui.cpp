#include "gui.h"
#include <iostream>


float ui_scale = 1.5f;                                  // scale value for the UI and UI text
bool RESET = false;
bool window_focused = true;
bool axis_change = true;                                // gets true when the axis plane is changes
extern int step;                                        // the steps between each glyph along all axis
extern int gui_VolumeSize[];
extern float     gui_PixelSize[];

int scroll_axis = 2;				                    // default axis is Z
int anisotropy = 0;                                     // 0: all tensors               1: linear tensors only
                                                        // 2: planar tensors only       3: spherical tensors only
float filter = 0.1f;
float zoom = 1.0f;
int cmap = 1;
float opacity = 1.0f;
float thresh = 0.0f;
float move[];
bool perspective = false;

bool OPEN_TENSOR = false;
bool RENDER_GLYPHS = false;
bool TENSOR_LOADED = false;
std::string TensorFileName;

bool OPEN_VOLUME = false;
bool RENDER_IMAGE = false;
bool VOLUME_LOADED = false;
std::string VolumeFileName;

bool tensor_data;
bool volume_data;
imgui_addons::ImGuiFileBrowser file_dialog;


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

void OpenFileDialog() {
    std::cout << "Loading \"" << file_dialog.selected_fn << "\" ..." << std::endl;
    if (tensor_data) {
        OPEN_TENSOR = true;
        TensorFileName = file_dialog.selected_path;
    }
    if (volume_data)
    {
        OPEN_VOLUME = true;
        VolumeFileName = file_dialog.selected_path;
    }

    tensor_data = false;
    volume_data = false;
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


    {
        // Use smaller font size
        float old_size = ImGui::GetFont()->Scale;
        ImGui::GetFont()->Scale *= 0.5;
        ImGui::PushFont(ImGui::GetFont());

        ImGui::Begin("Tensor");

        window_focused = (ImGui::IsWindowHovered()) ? false : true;

        // Change style 
        ImGuiStyle& style = ImGui::GetStyle();
        style.FrameRounding = 5.f;
        style.GrabRounding = 3.f;
        style.WindowRounding = 7.f;


        ////////////////////////////////////////////////  Load tensor field  ///////////////////////////////////////////////
        ImGui::SeparatorText("Load");
        if (ImGui::Button("Load Tensor"))					                                // create a button for loading the shader
        {
            ImGui::OpenPopup("Open File");
            tensor_data = true;
        }

        ImGui::SameLine();
        if (ImGui::Button("Render", ImVec2(60, 25)) && TENSOR_LOADED) {
            RENDER_GLYPHS = true;
        }

        ////////////////////////////////////////////////  Load volume  ////////////////////////////////////////////////////
        if (ImGui::Button("Load Volume"))					                                // create a button for loading the shader
        {
            ImGui::OpenPopup("Open File");
            volume_data = true;
        }
        ImGui::SameLine();
        // Render the plane with texture-mapped image
        ImGui::Checkbox("Image Plane", &RENDER_IMAGE);
        ImGui::Spacing();
        // Adjust the image transparency
        ImGui::SliderFloat("Opacity", &opacity, 0.1f, 1.0f);
        

        if (file_dialog.showFileDialog("Open File", imgui_addons::ImGuiFileBrowser::DialogMode::OPEN, ImVec2(700, 700), ".npy"))
            OpenFileDialog();
        ImGui::Spacing(); ImGui::Spacing();
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////


        // Select the number of tensors along X axis
        ImGui::SeparatorText("Pixel/Tensor");
        int smallest_axis = (gui_VolumeSize[0] < gui_VolumeSize[1]) ? gui_VolumeSize[0] : gui_VolumeSize[1];
        ImGui::Text("Number of pixels per tensor:");
        int test;
        if (TENSOR_LOADED)
            ImGui::SliderInt("  ", &step, 1, smallest_axis / 10);
        else {
            ImGui::SameLine();
            ImGui::Text("0");
        }
        ImGui::Spacing(); ImGui::Spacing();

        // Display the volume size (text only)
        ImGui::SeparatorText("Size");
        ImGui::DragInt3("Volume Size", gui_VolumeSize, 1, 0, 1500, "%d", ImGuiSliderFlags_NoInput);

        // Display and input the pixel size
        ImGui::DragFloat3("Pixel Size", gui_PixelSize, 0.001f, 0.f, 5.f, "%.3f");
        ImGui::Spacing();
        if (ImGui::SmallButton(" Reset ")) {
            gui_PixelSize[0] = 1.f;
            gui_PixelSize[1] = 1.f;
            gui_PixelSize[2] = 1.f;
        }
        ImGui::Spacing(); ImGui::Spacing();


        // Select which plane to render (view)
        /*ImGui::SeparatorText("Plane");
        if (ImGui::RadioButton("xi ", &scroll_axis, 0)) axis_change = true;
        ImGui::SameLine();
        if (ImGui::RadioButton("yi ", &scroll_axis, 1)) axis_change = true;
        ImGui::SameLine();
        if (ImGui::RadioButton("zi ", &scroll_axis, 2)) axis_change = true;
        ImGui::Spacing();*/

        // Filter anisotropy using a threshold filter option
        ImGui::SeparatorText("Anisotropy");
        ImGui::Combo(" ", &anisotropy, "All\0Linear\0Planar\0Spherical\0\0");
        ImGui::Spacing();
        ImGui::SliderFloat("Filter", &filter, 0.1f, 1.0f);
        ImGui::Spacing(); ImGui::Spacing();

        // Use perspective view instead of ortho view
        ImGui::SeparatorText("Projection");
        if (ImGui::RadioButton("Ortho", !perspective))
            perspective = false;
        ImGui::SameLine();
        if (ImGui::RadioButton("Perspective", perspective))
            perspective = true;
        ImGui::Spacing(); ImGui::Spacing();


        // Show colormapped image based on shortest/longest eigenvector
        ImGui::SeparatorText("IDK");
        ImGui::Combo("\t", &cmap, "LongestVector\0ShortestVector\0\0");
        ImGui::Spacing(); ImGui::Spacing();

        // Adjust a threshold for eigenvalues corresponding to each tensor
        ImGui::SeparatorText("Largest Eigenvalue Threshold");
        static float begin = 0.f, end = 125.f, step = 0.005f;
        ImGui::DragFloatRange2("Range", &begin, &end, 0.25f, 0.0f, 100, "Min: %.1f", "Max: %.1f");
        ImGui::DragFloat("Step", &step, 0.00001, 0.0f, 1.0f, "%.6f");
        ImGui::DragFloat("Threshold", &thresh, step, begin, end, "%.6f");
        ImGui::Spacing(); ImGui::Spacing();

        // Zooming in and out option
        ImGui::SeparatorText("Zoom");
        ImGui::InputFloat("", &zoom, 0.1f, 2.0f);
        zoom = (zoom < 1.0f) ? 1.0f : ((zoom > 5) ? 5 : zoom);
        ImGui::SameLine();
        if (ImGui::Button("O", ImVec2(25, 25))) zoom = 1.0f;             // reset zoom
        ImGui::Spacing(); ImGui::Spacing();

        // Reset button
        ImGui::SeparatorText("Reset");
        float avail = ImGui::GetContentRegionAvail().x;
        float off = (avail - 50) * 0.5f;
        if (off > 0.0f)
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + off);
        RESET = ImGui::Button("Reset", ImVec2(50, 50));

        ImGui::GetFont()->Scale = old_size;
        ImGui::PopFont();
        ImGui::End();
    }
    //ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);  // Render a separate window showing the FPS

    ImGui::Render();                                                            // Render all windows
}
#include "gui.h"
#include <iostream>


float ui_scale = 1.5f;                                  // scale value for the UI and UI text
bool RESET = false;
bool window_focused = true;
bool axis_change = true;                                // gets true when the axis plane is changes
extern int step;                                        // the steps between each glyph along all axis
extern int gui_VolumeSize[];
int scroll_axis = 2;				                    // default axis is Z
int anisotropy = 0;                                     // 0: all tensors               1: linear tensors only
                                                        // 2: planar tensors only       3: spherical tensors only
float filter = 0.1f;
float zoom = 1.0f;
int cmap = 2;
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
        if (ImGui::Button("Load Tensor"))					                                // create a button for loading the shader
        {
            ImGuiFileDialog::Instance()->OpenDialog("ChooseNpyFile", "Choose NPY File", ".npy,.npz", ".");
            tensor_data = true;
        }
        
        ImGui::SameLine();
        if (ImGui::Button("Render", ImVec2(60, 25)) && TENSOR_LOADED) {
            RENDER_GLYPHS = true;
        }
        
        ////////////////////////////////////////////////  Load volume  ////////////////////////////////////////////////////
        if (ImGui::Button("Load Volume"))					                                // create a button for loading the shader
        {
            ImGuiFileDialog::Instance()->OpenDialog("ChooseNpyFile", "Choose NPY File", ".npy,.npz", ".");
            volume_data = true;
        }
        ImGui::SameLine();
        // Render the plane with texture-mapped image
        ImGui::Checkbox("Image Plane", &RENDER_IMAGE);
        ImGui::Spacing();
        // Adjust the image transparency
        ImGui::SliderFloat("Opacity", &opacity, 0.1f, 1.0f);
        ImGui::Separator();

        if (ImGuiFileDialog::Instance()->Display("ChooseNpyFile")) {				    // if the user opened a file dialog
            if (ImGuiFileDialog::Instance()->IsOk())
            {
                std::string FileName = ImGuiFileDialog::Instance()->GetFilePathName();
                std::cout << "Loading \"" << FileName.substr(FileName.find_last_of("\\") + 1) << "\" ..." << std::endl;
                std::string extension = FileName.substr(FileName.find_last_of(".") + 1);

                if (extension == "npy")
                {
                    if (tensor_data) {
                        OPEN_TENSOR = true;
                        TensorFileName = FileName;
                    }
                    if (volume_data)
                    {
                        OPEN_VOLUME = true;
                        VolumeFileName = FileName;
                    }
                }
            }
            ImGuiFileDialog::Instance()->Close();
            tensor_data = false;
            volume_data = false;
        }

        // Select the number of tensors along X axis
        int smallest_axis = (gui_VolumeSize[0] < gui_VolumeSize[1]) ? gui_VolumeSize[0] : gui_VolumeSize[1];
        ImGui::Text("Number of pixels per tensor:");      
        int test;
        if (TENSOR_LOADED)
        {
            ImGui::SliderInt("", &step, 1, smallest_axis);
        }
        else
        {
            ImGui::SameLine();
            ImGui::Text("0");
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

        // Use perspective view instead of ortho view
        if (ImGui::RadioButton("Ortho", !perspective))
            perspective = false;
        ImGui::SameLine();
        if (ImGui::RadioButton("Perspective", perspective))
            perspective = true;
        ImGui::Separator();

        // Show colormapped image based on shortest/longest eigenvector
        ImGui::Columns(2);
        ImGui::Text("Colormap:");       ImGui::SameLine();
        ImGui::RadioButton("Shortest V", &cmap, 2);    ImGui::NextColumn();
        ImGui::RadioButton("Longest V", &cmap, 0);
        ImGui::Columns(1);
        ImGui::Spacing();

        // Adjust a threshold for eigenvalues corresponding to each tensor
        static float begin = 0.f, end = 125.f;
        //ImGui::PushItemWidth(50.f);
        //ImGui::InputFloat("", &begin);  ImGui::SameLine();
        //ImGui::PushItemWidth(200.f);
        //ImGui::DragFloat("", &thresh, 0.5f, begin, end);    
        ImGui::DragFloatRange2("Range", &begin, &end, 0.25f, 0.0f, 100, "Min: %.1f", "Max: %.1f");
        ImGui::DragFloat("Threshold", &thresh, 0.25f, begin, end);
        //ImGui::SameLine();
        //ImGui::PushItemWidth(50.f);
        //ImGui::InputFloat("", &end);
        //ImGui::PopItemWidth();
        ImGui::Separator();
        
        // Zooming in and out option
        ImGui::InputFloat("Zoom", &zoom, 0.1f, 2.0f);
        zoom = (zoom < 1.0f) ? 1.0f : ((zoom > 5) ? 5 : zoom);
        ImGui::SameLine();
        if(ImGui::Button("O", ImVec2(25, 25))) zoom = 1.0f;             // reset zoom

        ImGui::Separator();

        // Reset button
        RESET = ImGui::Button("Reset", ImVec2(70, 35));

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
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "ImGuiFileDialog/ImGuiFileDialog.h"

#include "tview2.h"
#include <tira/eigen.h>

ImVec4 clear_color = ImVec4(0.0f, 0.0f, 0.0f, 1.00f);   // specify the OpenGL color used to clear the back buffer

extern TV2_UI UI;

extern tira::image<glm::mat2> T0;
extern tira::image<glm::mat2> Tn;
extern tira::image<float> Scalar;
extern tira::image<float> Theta;
extern tira::image<float> Lambda;

/// <summary>
/// Initialize the GUI
/// </summary>
/// <param name="window">Pointer to the GLFW window that will be used for rendering</param>
/// <param name="glsl_version">Version of GLSL that will be used</param>
void ImGuiInit(GLFWwindow* window, const char* glsl_version) {
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    ImGui::GetStyle().ScaleAllSizes(UI.scale);
    ImGui::GetIO().FontGlobalScale = UI.scale;

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Load Fonts
    //io.Fonts->AddFontFromFileTTF("Roboto-Medium.ttf", ui_scale * 16.0f);

}

/// <summary>
/// Destroys the ImGui rendering interface (usually called when the program closes)
/// </summary>
void ImGuiDestroy() {
    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void ImGuiFieldSpecs() {
    ImGui::Text("Tensor Field Specifications");
    if (Tn.size() == 0) {
        ImGui::Text("None Loaded");
        return;
    }

    std::stringstream ss;
    ss << Tn.shape()[0] << " x " << Tn.shape()[1];

    ImGui::Text("%s", ("Field Size: " + ss.str()).c_str());
    //ImGui::Text("Maximum Norm: %f", MAXNORM);
}

/// re-calculate the scalar field based on the current settings in the UI
void RefreshScalarField() {
    switch (UI.scalar_type) {
    case ScalarType::EVal0:
        ImageFrom_Eigenvalue(&Lambda, &Scalar, 0);
        break;
    case ScalarType::EVal1:
        ImageFrom_Eigenvalue(&Lambda, &Scalar, 1);
        break;
    case ScalarType::Tensor00:
        ImageFrom_TensorElement2D(&Tn, &Scalar, 0, 0);
        break;
    case ScalarType::Tensor11:
        ImageFrom_TensorElement2D(&Tn, &Scalar, 1, 1);
        break;
    case ScalarType::Tensor01:
        ImageFrom_TensorElement2D(&Tn, &Scalar, 0, 1);
        break;
    case ScalarType::Eccentricity:
        ImageFrom_Eccentricity(&Lambda, &Scalar);
        break;
    case ScalarType::EVec0:
        ImageFrom_Theta(&Theta, &Scalar, 0);
        break;
    case ScalarType::EVec1:
        ImageFrom_Theta(&Theta, &Scalar, 1);
        break;

    default:
        throw std::runtime_error("Invalid scalar type");
    }
}

/// <summary>
/// Completely re-process the field from scratch. This is usually called when a new field is loaded
/// and processes the field based on the users previous settings.
/// </summary>
void ReprocessField() {

    if (UI.processing_type == ProcessingType::Gaussian)
        GaussianFilter(&T0, &Tn, UI.sigma, UI.cuda_device);
    else if (UI.processing_type == ProcessingType::Vote)
        TensorVote(&T0, &Tn, UI.sigma1, UI.vote_refinement, UI.sigma2, UI.stick_voting, UI.plate_voting, UI.cuda_device, UI.platevote_samples);
    else
        Tn = T0;
    EigenDecomposition(&Tn, &Lambda, &Theta, UI.cuda_device);
    RefreshScalarField();
    GenerateGlyphs();
    RefreshVisualization();
}

void RenderInspectionWindow() {

    int FieldIndex[2] = { static_cast<int>(UI.field_mouse_position[0]), static_cast<int>(UI.field_mouse_position[1]) };
    float f0 = UI.field_mouse_position[0];
    float f1 = UI.field_mouse_position[1];

    if (UI.field_loaded) {
        if ((f0 < 0 || f0 >= static_cast<int>(Tn.width()))
            && (f1 < 0 || f1 >= static_cast<int>(Tn.height())))
            ImGui::Text("Field Index: ---, ---");
        else if (f1 < 0 || f1 >= static_cast<int>(Tn.height()))
            ImGui::Text("Field Index: %d, ---", FieldIndex[0]);
        else if (f0 < 0 || f0 >= static_cast<int>(Tn.width()))
            ImGui::Text("Field Index: ---, %d", (int)f1);
        else {
            ImGui::Text("Field Index: %d, %d", (int)f0, (int)f1);


            // get the current tensor value
            glm::mat2 T = Tn((int)f0, (int)f1);

            // display the current tensor as a matrix
            ImGui::Text("Tensor:");
            float Row0[2] = { T[0][0], T[0][1] };
            ImGui::InputFloat2("##Row0", Row0, "%1.5F");
            float Row1[2] = { T[1][0], T[1][1] };
            ImGui::InputFloat2("##Row1", Row1, "%1.5F");

            // calculate the eigenvalues
            //glm::vec2 evals = Eigenvalues2D(T);
            float eval0, eval1;
            eval2D<float>((float*)&T, eval0, eval1);
            glm::vec2 evals(eval0, eval1);

            // display the eigenvalues
            ImGui::Text("Eigenvalues:");
            float lambdas[2] = { evals[0], evals[1] };
            ImGui::InputFloat2("##lambdas", lambdas, "%1.5e");
            ImGui::Text("Eccentricity:");
            ImGui::Columns(2);
            float e = Eccentricity2(lambdas[0], lambdas[1]);
            ImGui::InputFloat("e", &e, 0.0f, 0.0f, "%1.5e");
            //ImGui::SameLine();
            float c = LinearEccentricity2(lambdas[0], lambdas[1]);
            ImGui::InputFloat("c", &c, 0.0f, 0.0f, "%1.5e");

            ImGui::Columns(1);

            // calculate the eigenvectors
            float theta0, theta1;
            evec2Dpolar<float>((float*)(&T), (float*)(&evals), theta0, theta1);
            glm::vec2 ev0(std::cos(theta0), std::sin(theta0));
            glm::vec2 ev1(std::cos(theta1), std::sin(theta1));

            // display the eigenvectors
            ImGui::Text("Eigenvectors:");
            ImGui::Columns(2);
            ImGui::InputFloat2("x0, y0", reinterpret_cast<float*>(&ev0), "%1.5F");
            ImGui::InputFloat2("x1, y1", reinterpret_cast<float*>(&ev1), "%1.5F");
            ImGui::NextColumn();
            ImGui::InputFloat("theta0", &theta0);
            ImGui::InputFloat("theta1", &theta1);



        }
    }
}

/// This function renders the user interface every frame
void ImGuiRender() {
    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);  // Render a separate window showing the FPS
    ImGui::Text("File: %s", UI.loaded_filename == "" ? "N/A" : UI.loaded_filename.c_str());

    if (ImGui::Button("Load Field"))					// create a button for loading the shader
        ImGuiFileDialog::Instance()->OpenDialog("LoadNpyFile", "Choose NPY File", ".npy,.npz", ".");
    if (ImGuiFileDialog::Instance()->Display("LoadNpyFile")) {				    // if the user opened a file dialog
        if (ImGuiFileDialog::Instance()->IsOk()) {								    // and clicks okay, they've probably selected a file
            std::string filename = ImGuiFileDialog::Instance()->GetFilePathName();	// get the name of the file

            if (std::string extension = filename.substr(filename.find_last_of('.') + 1); extension == "npy") {
                UI.loaded_filename = filename;              // store the file name in the UI
                LoadTensorField(UI.loaded_filename, &T0);
                UI.field_loaded = true;
                Tn = T0;
                EigenDecomposition(&Tn, &Lambda, &Theta, UI.cuda_device);
                RefreshScalarField();
                RefreshVisualization();
            }
        }
        ImGuiFileDialog::Instance()->Close();									// close the file dialog box
    }
    ImGui::SameLine();
    if (ImGui::Button("Impulse")) UI.impulse_window = true;
    if (UI.impulse_window) {
        ImGui::Begin("Impulse");
        if (ImGui::InputInt("Resolution", &UI.impulse_resolution, 1)) {
            if (UI.field_impulse) {
                GenerateImpulse(&T0, UI.impulse_resolution, UI.impulse_theta, UI.impulse_anisotropy);
                ReprocessField();
            }
        }
        if (ImGui::SliderFloat("theta", &UI.impulse_theta, 0.0f, std::numbers::pi)) {
            if (UI.field_impulse) {
                GenerateImpulse(&T0, UI.impulse_resolution, UI.impulse_theta, UI.impulse_anisotropy);
                ReprocessField();
            }
        }
        if (ImGui::SliderFloat("anisotropy", &UI.impulse_anisotropy, 0.0f, 1.0f)) {
            if (UI.field_impulse) {
                GenerateImpulse(&T0, UI.impulse_resolution, UI.impulse_theta, UI.impulse_anisotropy);
                ReprocessField();
            }
        }
        if (!UI.field_impulse) {
            if (ImGui::Button("Impulse On")) {
                UI.field_impulse = true;
                GenerateImpulse(&T0, UI.impulse_resolution, UI.impulse_theta, UI.impulse_anisotropy);
                ReprocessField();
            }
        }
        else {
            if (ImGui::Button("Impulse Off")) {
                UI.field_impulse = false;
            }
        }
        if (ImGui::Button("Close")) UI.impulse_window = false;
        ImGui::End();
    }

    if (ImGui::Button("Save Field"))
        ImGuiFileDialog::Instance()->OpenDialog("SaveNpyFile", "Choose NPY File", ".npy,.npz", ".");
    if (ImGuiFileDialog::Instance()->Display("SaveNpyFile")) {				    // if the user opened a file dialog
        if (ImGuiFileDialog::Instance()->IsOk()) {								    // and clicks okay, they've probably selected a file
            std::string filename = ImGuiFileDialog::Instance()->GetFilePathName();	// get the name of the file

            if (std::string extension = filename.substr(filename.find_last_of('.') + 1); extension == "npy") {
                Tn.save_npy<float>(filename);
            }
        }
        ImGuiFileDialog::Instance()->Close();									// close the file dialog box
    }
    ImGuiFieldSpecs();


    const char* combo_preview_value = UI.device_names[UI.cuda_device + 1].c_str();
    if (ImGui::BeginCombo("CUDA Device", combo_preview_value))
    {
        for (int n = 0; n < UI.num_devices + 1; n++)
        {
            const bool is_selected = (UI.cuda_device + 1 == n);
            if (ImGui::Selectable(UI.device_names[n].c_str(), is_selected))
                UI.cuda_device = n - 1;

            // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
            if (is_selected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

    // select scalar component
    if (ImGui::Button("Reset View")) UI.camera_zoom = 1.0f;
    //float cam_pos[2] = { UI.camera_position[0], UI.camera_position[1] };
    ImGui::InputFloat2("Camera Position", (float*)&UI.camera_position);
    ImGui::InputFloat("Camera Zoom", &UI.camera_zoom);

    if (ImGui::Button("Save Image"))					// create a button for loading the shader
        ImGuiFileDialog::Instance()->OpenDialog("ChooseBmpFile", "Choose BMP File", ".bmp", ".");
    if (ImGuiFileDialog::Instance()->Display("ChooseBmpFile")) {				    // if the user opened a file dialog
        if (ImGuiFileDialog::Instance()->IsOk()) {								    // and clicks okay, they've probably selected a file
            std::string filename = ImGuiFileDialog::Instance()->GetFilePathName();	// get the name of the file
            if (std::string extension = filename.substr(filename.find_last_of('.') + 1); extension == "bmp") {
                //CurrentColormap.save(filename);
                throw std::runtime_error("Implement a way to save the colormapped image");
            }
        }
        ImGuiFileDialog::Instance()->Close();									// close the file dialog box
    }




    ImGui::SeparatorText("Scalar Display");
    ImGui::RadioButton("None", &UI.scalar_type, (int)ScalarType::NoScalar);
    if (ImGui::RadioButton("[0, 0] = dxdx", &UI.scalar_type, (int)ScalarType::Tensor00)) {
        //ScalarRefresh();
        ImageFrom_TensorElement2D(&Tn, &Scalar, 0, 0);
        UI.scalar_min = Scalar.minv();
        UI.scalar_max = Scalar.maxv();
        RefreshVisualization();
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("[1, 1] = dydy", &UI.scalar_type, (int)ScalarType::Tensor11)) {
        ImageFrom_TensorElement2D(&Tn, &Scalar, 1, 1);
        UI.scalar_min = Scalar.minv();
        UI.scalar_max = Scalar.maxv();
        RefreshVisualization();
    }
    if (ImGui::RadioButton("[0, 1] = dxdy", &UI.scalar_type, (int)ScalarType::Tensor01)) {
        ImageFrom_TensorElement2D(&Tn, &Scalar, 0, 1);
        UI.scalar_min = Scalar.minv();
        UI.scalar_max = Scalar.maxv();
        RefreshVisualization();
    }
    if (ImGui::RadioButton("lambda 0", &UI.scalar_type, (int)ScalarType::EVal0)) {
        ImageFrom_Eigenvalue(&Lambda, &Scalar, 0);
        UI.scalar_min = Scalar.minv();
        UI.scalar_max = Scalar.maxv();
        RefreshVisualization();
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("lambda 1", &UI.scalar_type, (int)ScalarType::EVal1)) {
        ImageFrom_Eigenvalue(&Lambda, &Scalar, 1);
        UI.scalar_min = Scalar.minv();
        UI.scalar_max = Scalar.maxv();
        RefreshVisualization();
    }

    if (ImGui::RadioButton("eccentricity", &UI.scalar_type, (int)ScalarType::Eccentricity)) {
        ImageFrom_Eccentricity(&Lambda, &Scalar);
        UI.scalar_min = Scalar.minv();
        UI.scalar_max = Scalar.maxv();
        RefreshVisualization();
    }
    if (ImGui::RadioButton("linear eccentricity", &UI.scalar_type, (int)ScalarType::LinearEccentricity)) {
        ImageFrom_LinearEccentricity(&Lambda, &Scalar);
        UI.scalar_min = Scalar.minv();
        UI.scalar_max = Scalar.maxv();
        RefreshVisualization();
    }
    if (ImGui::RadioButton("evec 0 (theta)", &UI.scalar_type, (int)ScalarType::EVec0)) {
        ImageFrom_Theta(&Theta, &Scalar, 0);
        UI.scalar_min = Scalar.minv();
        UI.scalar_max = Scalar.maxv();
        RefreshVisualization();
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("evec 1 (theta)", &UI.scalar_type, (int)ScalarType::EVec1)) {
        ImageFrom_Theta(&Theta, &Scalar, 1);
        UI.scalar_min = Scalar.minv();
        UI.scalar_max = Scalar.maxv();
        RefreshVisualization();
    }

    std::stringstream ss;
    ss << "Min: " << UI.scalar_min << "\t Max: " << UI.scalar_max;
    ImGui::Text("%s", ss.str().c_str());

    if (UI.scalar_type == ScalarType::EVec0 || UI.scalar_type == ScalarType::EVec1) {
        ImGui::SeparatorText("Eigenvector Display");
        if (ImGui::RadioButton("Negative Evals", &UI.signed_eigenvalues, -1)) {
            RefreshVisualization();
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Magnitude Evals", &UI.signed_eigenvalues, 0)) {
            RefreshVisualization();
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Positive Evals", &UI.signed_eigenvalues, 1)) {
            RefreshVisualization();
        }

        ImGui::Columns(2);
        ImGui::Text("Eccentricity");
        if (ImGui::RadioButton("None##1", &UI.eccentricity_color_mode, AdjustColorType::NoAdjustment))
            RefreshVisualization();
        if (ImGui::RadioButton("Lighten##1", &UI.eccentricity_color_mode, AdjustColorType::Lighten))
            RefreshVisualization();
        if (ImGui::RadioButton("Darken##1", &UI.eccentricity_color_mode, AdjustColorType::Darken))
            RefreshVisualization();

        ImGui::NextColumn();
        ImGui::Text("Magnitude");
        if (ImGui::RadioButton("None##2", &UI.magnitude_color_mode, AdjustColorType::NoAdjustment))
            RefreshVisualization();
        if (ImGui::RadioButton("Lighten##2", &UI.magnitude_color_mode, AdjustColorType::Lighten))
            RefreshVisualization();
        if (ImGui::RadioButton("Darken##2", &UI.magnitude_color_mode, AdjustColorType::Darken))
            RefreshVisualization();
        ImGui::Columns(1);
        ImGui::SetNextItemWidth(160);
        if (ImGui::InputFloat("Scale", &UI.magnitude_color_threshold, UI.largest_eigenvalue_magnitude * 0.01f, UI.largest_eigenvalue_magnitude * 0.1f)) {
            if (UI.magnitude_color_threshold < 0) UI.magnitude_color_threshold = 0;
            RefreshVisualization();
        }
        if (ImGui::Button("Reset")) {
            UI.magnitude_color_threshold = UI.largest_eigenvalue_magnitude;
            RefreshVisualization();
        }
    }

    if (ImGui::TreeNodeEx("Processing", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::Button("Bake")) {
            T0 = Tn;
            UI.processing_type = ProcessingType::NoProcessing;
        }
        if (ImGui::RadioButton("None", &UI.processing_type, (int)ProcessingType::NoProcessing)) {
            Tn = T0;
            EigenDecomposition(&Tn, &Lambda, &Theta, UI.cuda_device);
            RefreshScalarField();
            RefreshVisualization();
        }
        ImGui::SeparatorText("Gaussian Blur");
        if (ImGui::RadioButton("Blur", &UI.processing_type, (int)ProcessingType::Gaussian)) {
            if (UI.processing_type == ProcessingType::Gaussian) {
                GaussianFilter(&T0, &Tn, UI.sigma, UI.cuda_device);
                EigenDecomposition(&Tn, &Lambda, &Theta, UI.cuda_device);
                RefreshScalarField();
                GenerateGlyphs();                                               // the field size has changed, so regenerate glyphs
                RefreshVisualization();
            }
            else {
                Tn = T0;
                EigenDecomposition(&Tn, &Lambda, &Theta, UI.cuda_device);
                RefreshScalarField();
                GenerateGlyphs();                                               // the field size has changed, so regenerate glyphs
                RefreshVisualization();
            }
        }
        ImGui::SameLine();
        if (ImGui::InputFloat("##Sigma", &UI.sigma, 0.2f, 1.0f)) {
            if (UI.sigma <= 0) UI.sigma = 0.01;
            if (UI.processing_type == ProcessingType::Gaussian) {
                GaussianFilter(&T0, &Tn, UI.sigma, UI.cuda_device);
                EigenDecomposition(&Tn, &Lambda, &Theta, UI.cuda_device);
                RefreshScalarField();
                GenerateGlyphs();                                               // the field size has changed, so regenerate glyphs
                RefreshVisualization();
            }
        }

        ImGui::SeparatorText("Tensor Voting");
        if (ImGui::RadioButton("Tensor Voting", &UI.processing_type, (int)ProcessingType::Vote)) {
            TensorVote(&T0, &Tn, UI.sigma1, UI.vote_refinement, UI.sigma2, UI.stick_voting, UI.plate_voting, UI.cuda_device, UI.platevote_samples);
            EigenDecomposition(&Tn, &Lambda, &Theta, UI.cuda_device);
            RefreshScalarField();
            GenerateGlyphs();                                               // the field size has changed, so regenerate glyphs
            RefreshVisualization();
        }
        if (ImGui::InputFloat("Sigma 1", &UI.sigma1, 0.2f, 1.0f)) {
            if (UI.sigma1 < 0) UI.sigma1 = 0.0;
            if (UI.processing_type == ProcessingType::Vote) {
                TensorVote(&T0, &Tn, UI.sigma1, UI.vote_refinement, UI.sigma2, UI.stick_voting, UI.plate_voting, UI.cuda_device, UI.platevote_samples);
                EigenDecomposition(&Tn, &Lambda, &Theta, UI.cuda_device);
                RefreshScalarField();
                GenerateGlyphs();                                               // the field size has changed, so regenerate glyphs
                RefreshVisualization();
            }
        }
        if (ImGui::InputFloat("Sigma 2", &UI.sigma2, 0.2f, 1.0f)) {
            if (UI.sigma2 < 0) UI.sigma2 = 0.0;
            if (UI.processing_type == ProcessingType::Vote) {
                TensorVote(&T0, &Tn, UI.sigma1, UI.vote_refinement, UI.sigma2, UI.stick_voting, UI.plate_voting, UI.cuda_device, UI.platevote_samples);
                EigenDecomposition(&Tn, &Lambda, &Theta, UI.cuda_device);
                RefreshScalarField();
                GenerateGlyphs();                                               // the field size has changed, so regenerate glyphs
                RefreshVisualization();
            }
        }
        if (ImGui::InputInt("Power", &UI.vote_refinement, 1, 5)) {
            if (UI.vote_refinement < 1) UI.vote_refinement = 1;
            if (UI.processing_type == ProcessingType::Vote) {
                TensorVote(&T0, &Tn, UI.sigma1, UI.vote_refinement, UI.sigma2, UI.stick_voting, UI.plate_voting, UI.cuda_device, UI.platevote_samples);
                EigenDecomposition(&Tn, &Lambda, &Theta, UI.cuda_device);
                RefreshScalarField();
                GenerateGlyphs();                                               // the field size has changed, so regenerate glyphs
                RefreshVisualization();
            }
        }
        //ImGui::Columns(2);
        if (ImGui::Checkbox("Stick", &UI.stick_voting)) {
            if (UI.processing_type == ProcessingType::Vote) {
                TensorVote(&T0, &Tn, UI.sigma1, UI.vote_refinement, UI.sigma2, UI.stick_voting, UI.plate_voting, UI.cuda_device, UI.platevote_samples);
                EigenDecomposition(&Tn, &Lambda, &Theta, UI.cuda_device);
                RefreshScalarField();
                GenerateGlyphs();                                               // the field size has changed, so regenerate glyphs
                RefreshVisualization();
            }
        }
        ImGui::SameLine();
        if (ImGui::Checkbox("Plate", &UI.plate_voting)) {
            if (UI.processing_type == ProcessingType::Vote) {
                TensorVote(&T0, &Tn, UI.sigma1, UI.vote_refinement, UI.sigma2, UI.stick_voting, UI.plate_voting, UI.cuda_device, UI.platevote_samples);
                EigenDecomposition(&Tn, &Lambda, &Theta, UI.cuda_device);
                RefreshScalarField();
                GenerateGlyphs();                                               // the field size has changed, so regenerate glyphs
                RefreshVisualization();
            }
        }
        //ImGui::NextColumn();
        if (ImGui::InputInt("samples", &UI.platevote_samples)) {
            if (UI.processing_type == ProcessingType::Vote){
                if (UI.platevote_samples < 0) UI.platevote_samples = 0;
                TensorVote(&T0, &Tn, UI.sigma1, UI.vote_refinement, UI.sigma2, UI.stick_voting, UI.plate_voting, UI.cuda_device, UI.platevote_samples);
                EigenDecomposition(&Tn, &Lambda, &Theta, UI.cuda_device);
                RefreshScalarField();
                GenerateGlyphs();                                               // the field size has changed, so regenerate glyphs
                RefreshVisualization();
            }
        }
        //ImGui::Columns(1);

        ImGui::TreePop();
    }

    ImGui::Checkbox("Glyphs", &UI.render_glyphs);
    ImGui::InputFloat("Glyph Scale", &UI.glyph_scale, 0.1f, 1.0f);
    if (ImGui::InputInt("Tesselate", &UI.glyph_tesselation, 1, 10)) GenerateGlyphs();
    ImGui::Checkbox("Scale by Norm", &UI.glyph_normalize);

    

    if (!UI.inspection_window) {
        if (ImGui::Button("Open Inspection")) UI.inspection_window = true;
    }
    else {
        if (ImGui::Button("Close Inspection")) UI.inspection_window = false;
        ImGui::Begin("Inspection");
        RenderInspectionWindow();
        if (ImGui::Button("Close")) UI.inspection_window = false;
        ImGui::End();
    }

    ImGui::Render();                                                            // Render all windows
}

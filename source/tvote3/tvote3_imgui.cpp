#include "tvote3.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "../include/glyphs.h"
#include "../ImGuiFileDialog/ImGuiFileDialog.h"

#include <tira/graphics/glOrthoView.h>
#include <tira/eigen.h>

#include <glm/gtc/matrix_access.hpp>
//#include <Eigen/Core>
//#include <Eigen/Eigenvalues>
//#include <glm/gtc/type_ptr.hpp>

extern TV3_UI UI;

extern tira::volume<glm::mat3> T0;								// 3D tensor field (3x3 voxels)
extern tira::volume<glm::mat3> Tn;								// processed tensor field
extern tira::volume<float> Lambda;								// eigenvalues of the tensor field
extern tira::volume<glm::vec2> ThetaPhi;						// eigenvectors of the tensor field (in spherical coordinates)
extern tira::volume<float> Scalar;								// scalar field that is currently being visualized
extern tira::glOrthoView<unsigned char>* OrthoViewer;

static int voxel_coords[3] = { 0,0,0 };

/// re-calculate the scalar field based on the current settings in the UI
void UpdateScalarField() {
	switch (UI.scalar_type) {
	case ScalarType::EVal0:
		VolumeFrom_Eigenvalue(&Lambda, &Scalar, 0);
		break;
	case ScalarType::EVal1:
		VolumeFrom_Eigenvalue(&Lambda, &Scalar, 1);
		break;
	case ScalarType::EVal2:
		VolumeFrom_Eigenvalue(&Lambda, &Scalar, 2);
		break;
	case ScalarType::Tensor00:
		VolumeFrom_TensorElement3D(&Tn, &Scalar, 0, 0);
		break;
	case ScalarType::Tensor01:
		VolumeFrom_TensorElement3D(&Tn, &Scalar, 0, 1);
		break;
	case ScalarType::Tensor11:
		VolumeFrom_TensorElement3D(&Tn, &Scalar, 1, 1);
		break;
	case ScalarType::Tensor02:
		VolumeFrom_TensorElement3D(&Tn, &Scalar, 0, 2);
		break;
	case ScalarType::Tensor12:
		VolumeFrom_TensorElement3D(&Tn, &Scalar, 1, 2);
		break;
	case ScalarType::Tensor22:
		VolumeFrom_TensorElement3D(&Tn, &Scalar, 2, 2);
		break;
	case ScalarType::FractionalAnisotropy:
		throw std::runtime_error("Not implemented");
	case ScalarType::EVec0:										// if the eigenvectors are being display, there is no scalar field to update
		//throw std::runtime_error("Not implemented");
		break;
	case ScalarType::EVec1:
		//throw std::runtime_error("Not implemented");
		break;
	case ScalarType::EVec2:
		break;
	default:
		throw std::runtime_error("Invalid scalar type");
	}
	UpdateColormap();
	RefreshVisualization();
}

void ReprocessTensors() {
	if (UI.processing_type == ProcessingType::Gaussian)
		GaussianFilter(&T0, &Tn, UI.sigma, { T0.dx(), T0.dy(), T0.dz() }, UI.cuda_device);
	else if (UI.processing_type == ProcessingType::Vote)
		TensorVote(&T0, &Tn, UI.tv_sigma1, UI.tv_sigma2, UI.tv_power, UI.tv_stick, UI.tv_plate, UI.cuda_device, UI.platevote_samples);
	else
		Tn = T0;
}

void ReprocessField() {
	ReprocessTensors();
	EigenDecomposition(&Tn, &Lambda, &ThetaPhi, UI.cuda_device);
	UpdateScalarField();
}

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
	//ImGui::GetStyle().ScaleAllSizes(UI.scale);
	//ImGui::GetIO().FontGlobalScale = UI.scale;

	// Setup Platform/Renderer backends
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init(glsl_version);

	// Load Fonts
	//io.Fonts->AddFontFromFileTTF("Roboto-Medium.ttf", UI.scale * 16.0f);

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

void RenderImpulseWindow() {
	ImGui::Begin("Impulse");

	if (ImGui::InputInt("Pixels", &UI.impulse_resolution, 1)) {
		if (UI.impulse_field_active) {
			GenerateImpulseField(&T0, UI.impulse_resolution, UI.impulse_stick, UI.impulse_plate, UI.impulse_lambdas);
			//UI.field_loaded = true;
			ReprocessField();
		}
	}
	ImGui::SeparatorText("Stick Orientation (largest eigenvector)");
	ImGui::Columns(2);
	if (ImGui::SliderFloat("theta##stick", &UI.impulse_stick[0], 0, 2 * PI)) {
		if (UI.impulse_field_active) {
			GenerateImpulseField(&T0, UI.impulse_resolution, UI.impulse_stick, UI.impulse_plate, UI.impulse_lambdas);
			ReprocessField();
		}
	}

	ImGui::NextColumn();
	if (ImGui::SliderFloat("phi##stick", &UI.impulse_stick[1], 0, PI)) {
		if (UI.impulse_field_active) {
			GenerateImpulseField(&T0, UI.impulse_resolution, UI.impulse_stick, UI.impulse_plate, UI.impulse_lambdas);
			ReprocessField();
		}
	}
	ImGui::Columns(1);
	float cos_theta = std::cos(UI.impulse_stick[0]);
	float sin_theta = std::sin(UI.impulse_stick[0]);
	float cos_phi = std::cos(UI.impulse_stick[1]);
	float sin_phi = std::sin(UI.impulse_stick[1]);
	float cart_v2[3] = {cos_theta * sin_phi, sin_theta * sin_phi, cos_phi};
	ImGui::PushID(0);
	ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor(std::abs(cart_v2[0]), std::abs(cart_v2[1]), std::abs(cart_v2[2])));
	ImGui::InputFloat3("##cart_v2", &cart_v2[0]);
	ImGui::PopStyleColor();
	ImGui::PopID();

	ImGui::SeparatorText("Plate Orientation (smallest eigenvector)");
	if (ImGui::SliderFloat("theta##plate", &UI.impulse_plate, 0.0f, 2 * PI)) {
		if (UI.impulse_field_active) {
			GenerateImpulseField(&T0, UI.impulse_resolution, UI.impulse_stick, UI.impulse_plate, UI.impulse_lambdas);
			//UI.field_loaded = true;
			ReprocessField();
		}
	}
	ImGui::SeparatorText("Lambda");
	if (ImGui::SliderFloat("l1", &UI.impulse_lambdas[1], 0.0f, 1.0f)) {
		if (UI.impulse_lambdas[1] < UI.impulse_lambdas[0]) UI.impulse_lambdas[0] = UI.impulse_lambdas[1];

		if (UI.impulse_field_active) {
			GenerateImpulseField(&T0, UI.impulse_resolution, UI.impulse_stick, UI.impulse_plate, UI.impulse_lambdas);
			//UI.field_loaded = true;
			ReprocessField();
		}
	}
	if (ImGui::SliderFloat("l0", &UI.impulse_lambdas[0], 0.0f, 1.0f)) {
		if (UI.impulse_lambdas[1] < UI.impulse_lambdas[0]) UI.impulse_lambdas[1] = UI.impulse_lambdas[0];
		if (UI.impulse_field_active) {
			GenerateImpulseField(&T0, UI.impulse_resolution, UI.impulse_stick, UI.impulse_plate, UI.impulse_lambdas);
			//UI.field_loaded = true;
			ReprocessField();
		}
	}

	// calculate the impulse tensor so that it can be displayed in the window UI
	glm::mat3 P = GenerateImpulse(UI.impulse_stick, UI.impulse_plate, UI.impulse_lambdas);

	ImGui::SeparatorText("Impulse Tensor");
	glm::vec3 row1 = glm::row(P, 0);
	glm::vec3 row2 = glm::row(P, 1);
	glm::vec3 row3 = glm::row(P, 2);
	ImGui::InputFloat3("##row1", &row1[0]);
	ImGui::InputFloat3("##row2", &row2[0]);
	ImGui::InputFloat3("##row3", &row3[0]);


	float evals[3];
	tira::eval3_symmetric(P[0][0], P[1][0], P[1][1], P[2][0], P[2][1], P[2][2], evals[0], evals[1], evals[2]);

	float v0[3];
	float v1[3];
	float v2[3];
	tira::evec3_symmetric(P[0][0], P[1][0], P[1][1], P[2][0], P[2][1], P[2][2], evals, v0, v1, v2);

	ImGui::SeparatorText("Cartesian Eigenvectors");

	float v0_spherical[2];
	float v1_spherical[2];
	float v2_spherical[2];
	tira::evec3spherical_symmetric(P[0][0], P[1][0], P[1][1], P[2][0], P[2][1], P[2][2], evals, v0_spherical, v1_spherical, v2_spherical);

	ImGui::PushID(0);
	glm::vec3 color_v2 = ColormapEigenvector(2, evals[0], evals[1], evals[2], v2_spherical[0], v2_spherical[1]);
	ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor(color_v2.r, color_v2.g, color_v2.b));
	ImGui::InputFloat3("v2", &v2[0]);
	ImGui::PopStyleColor();
	ImGui::PopID();
	ImGui::SameLine();
	ImGui::InputFloat("#eval2", &evals[2]);

	ImGui::PushID(0);
	glm::vec3 color_v1 = ColormapEigenvector(1, evals[0], evals[1], evals[2], v1_spherical[0], v1_spherical[1]);
	ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor(color_v1.r, color_v1.g, color_v1.b));
	ImGui::InputFloat3("v1", &v1[0]);
	ImGui::PopStyleColor();
	ImGui::PopID();
	ImGui::SameLine();
	ImGui::InputFloat("#eval1", &evals[1]);
	
	ImGui::PushID(0);
	glm::vec3 color_v0 = ColormapEigenvector(0, evals[0], evals[1], evals[2], v0_spherical[0], v0_spherical[1]);
	ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor(color_v0.r, color_v0.g, color_v0.b));
	ImGui::InputFloat3("v0", &v0[0]);
	ImGui::PopStyleColor();
	ImGui::PopID();
	ImGui::SameLine();
	ImGui::InputFloat("#eval0", &evals[0]);

	if (!UI.impulse_field_active) {
		if (ImGui::Button("Impulse On")) {
			UI.impulse_field_active = true;
			GenerateImpulseField(&T0, UI.impulse_resolution, UI.impulse_stick, UI.impulse_plate, UI.impulse_lambdas);
			UI.field_loaded = true;
			ReprocessField();
		}
	}
	else {
		if (ImGui::Button("Impulse Off")) {
			UI.impulse_field_active = false;
		}
	}
	if (ImGui::Button("Close")) UI.impulse_window = false;
	ImGui::End();
}

/// <summary>
/// This function renders the user interface every frame
/// </summary>
void ImGuiRender() {

	// Start the Dear ImGui frame
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	{
		ImGui::Begin("Tensor");
		UI.window_focused = (ImGui::IsWindowHovered()) ? false : true;

		if (ImGui::BeginTabBar("MyTabBar"))
		{
			if (ImGui::BeginTabItem("Tensor Field"))
			{
				// Change style 
				ImGuiStyle& style = ImGui::GetStyle();
				style.FrameRounding = 5.f;
				style.GrabRounding = 3.f;
				style.WindowRounding = 7.f;

				if (ImGui::Button("Impulse")) UI.impulse_window = true;
				if (UI.impulse_window) RenderImpulseWindow();

				ImGui::SameLine();

				if (ImGui::Button("Load Field"))					                        // create a button for loading the field
					ImGuiFileDialog::Instance()->OpenDialog("LoadNpyFile", "Choose NPY File", ".npy,.npz");
				if (ImGuiFileDialog::Instance()->Display("LoadNpyFile")) {				    // if the user opened a file dialog
					if (ImGuiFileDialog::Instance()->IsOk()) {								// and clicks okay, they've probably selected a file
						const std::string filename = ImGuiFileDialog::Instance()->GetFilePathName();	// get the name of the file

						if (const std::string extension = filename.substr(filename.find_last_of('.') + 1); extension == "npy") {
							T0.load_npy<float>(filename);
							ReprocessField();
						}
					}
					ImGuiFileDialog::Instance()->Close();									// close the file dialog box
				}

				ImGui::SameLine();

				if (ImGui::Button("Save Field"))
					ImGuiFileDialog::Instance()->OpenDialog("SaveNpyFile", "Choose NPY File", ".npy,.npz");
				if (ImGuiFileDialog::Instance()->Display("SaveNpyFile")) {				    // if the user opened a file dialog
					if (ImGuiFileDialog::Instance()->IsOk()) {								    // and clicks okay, they've probably selected a file
						const std::string filename = ImGuiFileDialog::Instance()->GetFilePathName();	// get the name of the file

						if (const std::string extension = filename.substr(filename.find_last_of('.') + 1); extension == "npy") {
							std::vector<size_t> output_shape = Tn.Shape();
							output_shape[3] = 3;
							output_shape.push_back(3);
							Tn.SaveNpy<float>(filename, output_shape);
						}
					}
					ImGuiFileDialog::Instance()->Close();									// close the file dialog box
				}

				ImGui::SameLine();
				if (ImGui::Button("Save Glyphs"))
					ImGuiFileDialog::Instance()->OpenDialog("SaveGlyphFile", "Choose an OBJ File", ".obj");
				if (ImGuiFileDialog::Instance()->Display("SaveGlyphFile")) {				    // if the user opened a file dialog
					if (ImGuiFileDialog::Instance()->IsOk()) {								    // and clicks okay, they've probably selected a file
						const std::string filename = ImGuiFileDialog::Instance()->GetFilePathName();	// get the name of the file

						if (const std::string extension = filename.substr(filename.find_last_of('.') + 1); extension == "obj") {
							field2glyphs(Tn, filename);
						}
					}
					ImGuiFileDialog::Instance()->Close();									// close the file dialog box
				}

				ImGui::SeparatorText("Slice Positions");
				glm::vec3 field_size = OrthoViewer->dimensions();
				if (ImGui::SliderFloat("X", &UI.slice_positions[0], 0.0f, field_size[0])) {
					RefreshVisualization();
				}
				if (ImGui::SliderFloat("Y", &UI.slice_positions[1], 0.0f, field_size[1])) {
					RefreshVisualization();
				}
				if (ImGui::SliderFloat("Z", &UI.slice_positions[2], 0.0f, field_size[2])) {
					RefreshVisualization();
				}

				ImGui::SeparatorText("Color Mapping");
				ImGui::RadioButton("None", &UI.scalar_type, (int)ScalarType::NoScalar);
				ImGui::SameLine();
				if (ImGui::RadioButton("Fractional Anisotropy", &UI.scalar_type, (int)ScalarType::FractionalAnisotropy)) {

				}
				ImGui::Columns(2);
				ImGui::SeparatorText("Tensor Scalars");
				if (ImGui::RadioButton("##dxdx", &UI.scalar_type, (int)ScalarType::Tensor00)) {
					UpdateScalarField();
				}
				ImGui::SameLine();
				if (ImGui::RadioButton("##dxdy", &UI.scalar_type, (int)ScalarType::Tensor01)) {
					UpdateScalarField();
				}
				ImGui::SameLine();
				if (ImGui::RadioButton("##dxdz", &UI.scalar_type, (int)ScalarType::Tensor02)) {
					UpdateScalarField();
				}
				if (ImGui::RadioButton("##dydx", &UI.scalar_type, (int)ScalarType::Tensor01)) {
					UpdateScalarField();
				}
				ImGui::SameLine();
				if (ImGui::RadioButton("##dydy", &UI.scalar_type, (int)ScalarType::Tensor11)) {
					UpdateScalarField();
				}
				ImGui::SameLine();
				if (ImGui::RadioButton("##dydz", &UI.scalar_type, (int)ScalarType::Tensor12)) {
					UpdateScalarField();
				}

				if (ImGui::RadioButton("##dzdx", &UI.scalar_type, (int)ScalarType::Tensor02)) {
					UpdateScalarField();
				}
				ImGui::SameLine();
				if (ImGui::RadioButton("##dzdy", &UI.scalar_type, (int)ScalarType::Tensor12)) {
					UpdateScalarField();
				}
				ImGui::SameLine();
				if (ImGui::RadioButton("##dzdz", &UI.scalar_type, (int)ScalarType::Tensor22)) {
					UpdateScalarField();
				}

				ImGui::NextColumn();

				ImGui::SeparatorText("Eigen");

				if (ImGui::RadioButton("l2", &UI.scalar_type, (int)ScalarType::EVal2)) {
					UpdateScalarField();
				}
				ImGui::SameLine();
				if (ImGui::RadioButton("evec2", &UI.scalar_type, (int)ScalarType::EVec2)) {
					UpdateColormap();
				}

				if (ImGui::RadioButton("l1", &UI.scalar_type, (int)ScalarType::EVal1)) {
					UpdateScalarField();
				}
				ImGui::SameLine();
				if (ImGui::RadioButton("evec1", &UI.scalar_type, (int)ScalarType::EVec1)) {
					UpdateColormap();
				}

				if (ImGui::RadioButton("l0", &UI.scalar_type, (int)ScalarType::EVal0)) {
					UpdateScalarField();
				}
				ImGui::SameLine();
				if (ImGui::RadioButton("evec0", &UI.scalar_type, (int)ScalarType::EVec0)) {
					UpdateColormap();
				}
				ImGui::Columns(1);

				ImGui::Separator();

				// Toggle button for device selection: GPU (0) <-> CPU (-1)
				ImGui::Text("Device:");
				ImGui::SameLine();
				bool isGPU = (UI.cuda_device == 0);
				if (ImGui::RadioButton("GPU", isGPU)) {
					UI.cuda_device = 0;
					ReprocessField();
				}
				ImGui::SameLine();
				if (ImGui::RadioButton("CPU", !isGPU)) {
					UI.cuda_device = -1;
					ReprocessField();
				}

				ImGui::EndTabItem();
			}
			
			if (ImGui::BeginTabItem("Processing"))
			{
				if (ImGui::Button("Bake")) {
					T0 = Tn;
					UI.processing_type = ProcessingType::NoProcessing;
				}
				if (ImGui::RadioButton("None", &UI.processing_type, (int)ProcessingType::NoProcessing)) {
					Tn = T0;
					EigenDecomposition(&Tn, &Lambda, &ThetaPhi, UI.cuda_device);
					UpdateScalarField();
					RefreshVisualization();
				}

				///////////////////////////////////////////////  Gaussian Blur  ///////////////////////////////////////////////////
				ImGui::SeparatorText("Gaussian Blur");
				if (ImGui::RadioButton("Blur", &UI.processing_type, (int)ProcessingType::Gaussian)) {
					if (UI.processing_type == ProcessingType::Gaussian) {
						GaussianFilter(&T0, &Tn, UI.sigma, { T0.dx(), T0.dy(), T0.dz() }, UI.cuda_device);
						EigenDecomposition(&Tn, &Lambda, &ThetaPhi, UI.cuda_device);
						UpdateScalarField();
						RefreshVisualization();
					}
					else {
						Tn = T0;
						EigenDecomposition(&Tn, &Lambda, &ThetaPhi, UI.cuda_device);
						UpdateScalarField();
						RefreshVisualization();
					}
				}
				ImGui::SameLine();
				if (ImGui::InputFloat("##Sigma", &UI.sigma, 0.2f, 1.0f)) {
					if (UI.sigma <= 0.0f) UI.sigma = 0.01f;
					float maxSigma0 = T0.X() * T0.dx() / 6.0f;
					float maxSigma1 = T0.Y() * T0.dy() / 6.0f;
					float maxSigma2 = T0.Z() * T0.dz() / 6.0f;
					float maxSigma = std::min({ maxSigma0, maxSigma1, maxSigma2 });
					if (UI.sigma >= maxSigma)	UI.sigma = maxSigma;
					if (UI.processing_type == ProcessingType::Gaussian) {
						GaussianFilter(&T0, &Tn, UI.sigma, { T0.dx(), T0.dy(), T0.dz() }, UI.cuda_device);
						EigenDecomposition(&Tn, &Lambda, &ThetaPhi, UI.cuda_device);
						UpdateScalarField();
						RefreshVisualization();
					}
				}

				///////////////////////////////////////////////  Tensor Voting  ///////////////////////////////////////////////////
				ImGui::SeparatorText("Tensor Vote");
				if (ImGui::RadioButton("Tensor Voting", &UI.processing_type, (int)ProcessingType::Vote)) {
					TensorVote(&T0, &Tn, UI.tv_sigma1, UI.tv_sigma2, UI.tv_power, UI.tv_stick, UI.tv_plate, UI.cuda_device, UI.platevote_samples);
					EigenDecomposition(&Tn, &Lambda, &ThetaPhi, UI.cuda_device);
					UpdateScalarField();
					RefreshVisualization();
				}

				if (ImGui::InputFloat("Sigma 1", &UI.tv_sigma1, 0.2f, 1.0f) && UI.processing_type == ProcessingType::Vote) {
					UI.tv_sigma1 = (UI.tv_sigma1 < 0.0f) ? 0.0f : UI.tv_sigma1;
					if (ImGui::IsItemEdited()) {
						TensorVote(&T0, &Tn, UI.tv_sigma1, UI.tv_sigma2, UI.tv_power, UI.tv_stick, UI.tv_plate, UI.cuda_device, UI.platevote_samples);
						EigenDecomposition(&Tn, &Lambda, &ThetaPhi, UI.cuda_device);
						UpdateScalarField();
						RefreshVisualization();
					}
				}

				if (ImGui::InputFloat("Sigma 2", &UI.tv_sigma2, 0.2f, 1.0f) && UI.processing_type == ProcessingType::Vote) {
					UI.tv_sigma2 = (UI.tv_sigma2 < 0.0f) ? 0.0f : UI.tv_sigma2;
					if (ImGui::IsItemEdited()) {
						TensorVote(&T0, &Tn, UI.tv_sigma1, UI.tv_sigma2, UI.tv_power, UI.tv_stick, UI.tv_plate, UI.cuda_device, UI.platevote_samples);
						EigenDecomposition(&Tn, &Lambda, &ThetaPhi, UI.cuda_device);
						UpdateScalarField();
						RefreshVisualization();
					}
				}

				if (ImGui::InputInt("Power", &UI.tv_power, 1, 5) && UI.processing_type == ProcessingType::Vote) {
					UI.tv_power = (UI.tv_power < 1) ? 1 : UI.tv_power;
					if (ImGui::IsItemEdited()) {
						TensorVote(&T0, &Tn, UI.tv_sigma1, UI.tv_sigma2, UI.tv_power, UI.tv_stick, UI.tv_plate, UI.cuda_device, UI.platevote_samples);
						EigenDecomposition(&Tn, &Lambda, &ThetaPhi, UI.cuda_device);
						UpdateScalarField();
						RefreshVisualization();
					}
				}

				if (ImGui::Checkbox("Stick", &UI.tv_stick) && UI.processing_type == ProcessingType::Vote)
					if (ImGui::IsItemEdited()) {
						TensorVote(&T0, &Tn, UI.tv_sigma1, UI.tv_sigma2, UI.tv_power, UI.tv_stick, UI.tv_plate, UI.cuda_device, UI.platevote_samples);
						EigenDecomposition(&Tn, &Lambda, &ThetaPhi, UI.cuda_device);
						UpdateScalarField();
						RefreshVisualization();
					}
				ImGui::SameLine();
				if (ImGui::Checkbox("Plate", &UI.tv_plate) && UI.processing_type == ProcessingType::Vote)
					if (ImGui::IsItemEdited()) {
						if (!UI.tv_plate) UI.platevote_samples = 0;			// reset to 0 when disabling plate voting

						TensorVote(&T0, &Tn, UI.tv_sigma1, UI.tv_sigma2, UI.tv_power, UI.tv_stick, UI.tv_plate, UI.cuda_device, UI.platevote_samples);
						EigenDecomposition(&Tn, &Lambda, &ThetaPhi, UI.cuda_device);
						UpdateScalarField();
						RefreshVisualization();
					}

				bool useNumerical = (UI.platevote_samples > 0);
				if (UI.tv_plate) {
					if (ImGui::Checkbox("Numerical", &useNumerical)) {
						if (!useNumerical) UI.platevote_samples = 0;								// reset to 0 when switching to analytical
						else UI.platevote_samples = 10;												// set to default when switching to numerical

						if (ImGui::IsItemEdited() && UI.processing_type == ProcessingType::Vote) {
							TensorVote(&T0, &Tn, UI.tv_sigma1, UI.tv_sigma2, UI.tv_power, UI.tv_stick, UI.tv_plate, UI.cuda_device, UI.platevote_samples);
							EigenDecomposition(&Tn, &Lambda, &ThetaPhi, UI.cuda_device);
							UpdateScalarField();
							RefreshVisualization();
						}
					}
					if (useNumerical) {
						if (ImGui::InputInt("Samples", &UI.platevote_samples))	{
							if (UI.platevote_samples < 0) UI.platevote_samples = 0;

							if (ImGui::IsItemEdited() && UI.processing_type == ProcessingType::Vote) {
								TensorVote(&T0, &Tn, UI.tv_sigma1, UI.tv_sigma2, UI.tv_power, UI.tv_stick, UI.tv_plate, UI.cuda_device, UI.platevote_samples);
								EigenDecomposition(&Tn, &Lambda, &ThetaPhi, UI.cuda_device);
								UpdateScalarField();
								RefreshVisualization();
							}
						}
					}
				}
				ImGui::EndTabItem();
			}

			if (ImGui::BeginTabItem("Voxel Inspector")) {
				ImGui::Text("Voxel Coordinate");

				if (Tn.Size() == 0 || Lambda.Size() == 0 || ThetaPhi.Size() == 0)
					ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "No field loaded");
				else {
					ImGui::SliderInt("X", &voxel_coords[0], 0, Tn.X() - 1);
					ImGui::SliderInt("Y", &voxel_coords[1], 0, Tn.Y() - 1);
					ImGui::SliderInt("Z", &voxel_coords[2], 0, Tn.Z() - 1);

					ImGui::Separator();

					int x = voxel_coords[0], y = voxel_coords[1], z = voxel_coords[2];
					glm::mat3 tensor = Tn(x, y, z);
					float evals[3] = { Lambda(x, y, z, 0), Lambda(x, y, z, 1), Lambda(x, y, z, 2) };

					// Get spherical eigenvectors
					glm::vec2 tp0 = ThetaPhi(x, y, z, 0);
					glm::vec2 tp1 = ThetaPhi(x, y, z, 1);
					glm::vec2 tp2 = ThetaPhi(x, y, z, 2);

					glm::vec3 v0 = { std::cos(tp0.x) * std::sin(tp0.y), std::sin(tp0.x) * std::sin(tp0.y), std::cos(tp0.y) };
					glm::vec3 v1 = { std::cos(tp1.x) * std::sin(tp1.y), std::sin(tp1.x) * std::sin(tp1.y), std::cos(tp1.y) };
					glm::vec3 v2 = { std::cos(tp2.x) * std::sin(tp2.y), std::sin(tp2.x) * std::sin(tp2.y), std::cos(tp2.y) };

					ImGui::Text("Tensor (column-major):");
					// glm::mat3 is column-major. We need a non-const copy for ImGui
					glm::mat3 display_tensor = tensor;
					ImGui::InputFloat3("Col 0", &display_tensor[0][0], "%.4f", ImGuiInputTextFlags_ReadOnly);
					ImGui::InputFloat3("Col 1", &display_tensor[1][0], "%.4f", ImGuiInputTextFlags_ReadOnly);
					ImGui::InputFloat3("Col 2", &display_tensor[2][0], "%.4f", ImGuiInputTextFlags_ReadOnly);

					ImGui::Separator();
					ImGui::SeparatorText("Eigenvalues & Eigenvectors");

					if (ImGui::BeginTable("EigTable", 2, ImGuiTableFlags_BordersInnerV))
					{
						// Setup columns
						ImGui::TableSetupColumn("Eigenvector", ImGuiTableColumnFlags_WidthStretch);
						ImGui::TableSetupColumn("Eigenvalue", ImGuiTableColumnFlags_WidthFixed, 100.0f);

						// v2 (Stick)
						ImGui::TableNextRow();
						ImGui::TableSetColumnIndex(0); // --- Column 0 ---
						ImGui::PushID(0);
						ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor(std::abs(v2.x), std::abs(v2.y), std::abs(v2.z)));
						ImGui::InputFloat3("EigVec 2 (Stick)", &v2[0], "%.4f", ImGuiInputTextFlags_ReadOnly);
						ImGui::PopStyleColor();
						ImGui::PopID();

						ImGui::TableSetColumnIndex(1); // --- Column 1 ---
						ImGui::InputFloat("##Eval2", &evals[2], 0, 0, "%.4f", ImGuiInputTextFlags_ReadOnly);

						// v1 (Plate)
						ImGui::TableNextRow();
						ImGui::TableSetColumnIndex(0); // --- Column 0 ---
						ImGui::PushID(1);
						ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor(std::abs(v1.x), std::abs(v1.y), std::abs(v1.z)));
						ImGui::InputFloat3("EigVec 1 (Plate)", &v1[0], "%.4f", ImGuiInputTextFlags_ReadOnly);
						ImGui::PopStyleColor();
						ImGui::PopID();

						ImGui::TableSetColumnIndex(1); // --- Column 1 ---
						ImGui::InputFloat("##Eval1", &evals[1], 0, 0, "%.4f", ImGuiInputTextFlags_ReadOnly);

						// v0 (Ball)
						ImGui::TableNextRow();
						ImGui::TableSetColumnIndex(0); // --- Column 0 ---
						ImGui::PushID(2);
						ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor(std::abs(v0.x), std::abs(v0.y), std::abs(v0.z)));
						ImGui::InputFloat3("EigVec 0 (Ball) ", &v0[0], "%.4f", ImGuiInputTextFlags_ReadOnly);
						ImGui::PopStyleColor();
						ImGui::PopID();

						ImGui::TableSetColumnIndex(1); // --- Column 1 ---
						ImGui::InputFloat("##Eval0", &evals[0], 0, 0, "%.4f", ImGuiInputTextFlags_ReadOnly);

						ImGui::EndTable();
					}
				}
				ImGui::EndTabItem();
			}
		ImGui::EndTabBar();
		}

		//ImGui::GetFont()->Scale = old_size;
		//ImGui::PopFont();
		ImGui::End();


	}
	//ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);  // Render a separate window showing the FPS

	ImGui::Render();                                                            // Render all windows

}
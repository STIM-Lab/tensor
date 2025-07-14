#include "tvote3.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "ImGuiFileDialog/ImGuiFileDialog.h"

#include <tira/graphics/glOrthoView.h>
#include <tira/eigen.h>

#include <glm/gtc/matrix_access.hpp>

extern TV3_UI UI;

extern tira::volume<glm::mat3> T0;								// 3D tensor field (3x3 voxels)
extern tira::volume<glm::mat3> Tn;								// processed tensor field
extern tira::volume<float> Lambda;								// eigenvalues of the tensor field
extern tira::volume<glm::vec2> ThetaPhi;						// eigenvectors of the tensor field (in spherical coordinates)
extern tira::volume<float> Scalar;								// scalar field that is currently being visualized
extern tira::glOrthoView<unsigned char>* OrthoViewer;

#define PI 3.14159265358979323846264338327950288

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
		break;
	case ScalarType::EVec0:
		throw std::runtime_error("Not implemented");
		break;
	case ScalarType::EVec1:
		throw std::runtime_error("Not implemented");
		break;
	default:
		throw std::runtime_error("Invalid scalar type");
	}
	UpdateColormap();
	RefreshVisualization();
}

void ReprocessField() {

	if (UI.processing_type == ProcessingType::Gaussian)
		GaussianFilter(&T0, &Tn, UI.sigma, { T0.dx(), T0.dy(), T0.dz() }, UI.cuda_device);
	else if (UI.processing_type == ProcessingType::Vote)
		//TensorVote(&T0, &Tn, UI.sigma1, UI.vote_refinement, UI.sigma2, UI.stick_voting, UI.plate_voting, UI.cuda_device, UI.platevote_samples);
		throw std::runtime_error("Not implemented");
	else
		Tn = T0;
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
	ImGui::GetStyle().ScaleAllSizes(UI.scale);
	ImGui::GetIO().FontGlobalScale = UI.scale;

	// Setup Platform/Renderer backends
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init(glsl_version);

	// Load Fonts
	io.Fonts->AddFontFromFileTTF("Roboto-Medium.ttf", UI.scale * 16.0f);

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

	glm::mat3 P = GenerateImpulse(UI.impulse_stick, UI.impulse_plate, UI.impulse_lambdas);
	float evals[3];
	tira::eval3_symmetric(P[0][0], P[1][0], P[1][1], P[2][0], P[2][1], P[2][2], evals[0], evals[1], evals[2]);
	//std::cout<<"Eigenvalues: "<<evals[0]<<", "<<eval1<<", "<<eval2<<std::endl;

	float evec0[3];
	float evec1[3];
	float evec2[3];
	tira::evec3_symmetric(P[0][0], P[1][0], P[1][1], P[2][0], P[2][1], P[2][2], evals, evec0, evec1, evec2);
	std::cout<<"Eigenvector 2: "<<evec2[0]<<", "<<evec2[1]<<", "<<evec2[2]<<std::endl;

	ImGui::Columns(2);
	if (ImGui::InputInt("Pixels", &UI.impulse_resolution, 1)) {
		if (UI.impulse_field) {
			GenerateImpulseField(&T0, UI.impulse_resolution, UI.impulse_stick, UI.impulse_plate, UI.impulse_lambdas);
			UI.field_loaded = true;
			ReprocessField();
		}
	}
	ImGui::Columns(1);
	ImGui::SeparatorText("Stick Orientation");
	ImGui::Columns(2);
	if (ImGui::SliderFloat("theta##stick", &UI.impulse_stick[0], 0, 2 * PI)) {
		if (UI.impulse_field) {
			GenerateImpulseField(&T0, UI.impulse_resolution, UI.impulse_stick, UI.impulse_plate, UI.impulse_lambdas);
			UI.field_loaded = true;
			ReprocessField();
		}
	}
	ImGui::NextColumn();
	if (ImGui::SliderFloat("phi##stick", &UI.impulse_stick[1], 0, PI)) {
		if (UI.impulse_field) {
			GenerateImpulseField(&T0, UI.impulse_resolution, UI.impulse_stick, UI.impulse_plate, UI.impulse_lambdas);
			UI.field_loaded = true;
			ReprocessField();
		}
	}
	ImGui::Columns(1);
	ImGui::SeparatorText("Plate Orientation");
	ImGui::Columns(2);
	//ImGui::NextColumn();
	if (ImGui::SliderFloat("theta##plate", &UI.impulse_plate, 0.0f, 2 * PI)) {
		if (UI.impulse_field) {
			GenerateImpulseField(&T0, UI.impulse_resolution, UI.impulse_stick, UI.impulse_plate, UI.impulse_lambdas);
			UI.field_loaded = true;
			ReprocessField();
		}
	}
	ImGui::Columns(1);
	ImGui::SeparatorText("Lambda");
	if (ImGui::SliderFloat("l1", &UI.impulse_lambdas[1], 0.0f, 1.0f)) {
		if (UI.impulse_lambdas[1] < UI.impulse_lambdas[0]) UI.impulse_lambdas[0] = UI.impulse_lambdas[1];

		if (UI.impulse_field) {
			GenerateImpulseField(&T0, UI.impulse_resolution, UI.impulse_stick, UI.impulse_plate, UI.impulse_lambdas);
			UI.field_loaded = true;
			ReprocessField();
		}
	}
	if (ImGui::SliderFloat("l0", &UI.impulse_lambdas[0], 0.0f, 1.0f)) {
		if (UI.impulse_lambdas[1] < UI.impulse_lambdas[0]) UI.impulse_lambdas[1] = UI.impulse_lambdas[0];
		if (UI.impulse_field) {
			GenerateImpulseField(&T0, UI.impulse_resolution, UI.impulse_stick, UI.impulse_plate, UI.impulse_lambdas);
			UI.field_loaded = true;
			ReprocessField();
		}
	}
	ImGui::SeparatorText("Impulse Tensor");
	glm::vec3 row1 = glm::row(P, 0);
	glm::vec3 row2 = glm::row(P, 1);
	glm::vec3 row3 = glm::row(P, 2);
	ImGui::InputFloat3("##row1", &row1[0]);
	ImGui::InputFloat3("##row2", &row2[0]);
	ImGui::InputFloat3("##row3", &row3[0]);

	ImGui::SeparatorText("Cartesian Eigenvectors");

	float cos_t2 = std::cos(UI.impulse_stick[0]);
	float sin_t2 = std::sin(UI.impulse_stick[0]);
	float cos_p2 = std::cos(UI.impulse_stick[1]);
	float sin_p2 = std::sin(UI.impulse_stick[1]);
	glm::vec3 v2 = glm::vec3(cos_t2*sin_p2, sin_t2*sin_p2, cos_p2);

	ImGui::PushID(0);
	ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor(std::abs(v2[0]), std::abs(v2[1]), std::abs(v2[2])));
	ImGui::InputFloat3("##row1", &v2[0]);
	ImGui::PopStyleColor();
	ImGui::PopID();
	


	if (!UI.impulse_field) {
		if (ImGui::Button("Impulse On")) {
			UI.impulse_field = true;
			GenerateImpulseField(&T0, UI.impulse_resolution, UI.impulse_stick, UI.impulse_plate, UI.impulse_lambdas);
			UI.field_loaded = true;
			ReprocessField();
		}
	}
	else {
		if (ImGui::Button("Impulse Off")) {
			UI.impulse_field = false;
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
		// Use smaller font size
		float old_size = ImGui::GetFont()->Scale;
		ImGui::GetFont()->Scale *= 0.5;
		ImGui::PushFont(ImGui::GetFont());

		ImGui::Begin("Tensor");


		UI.window_focused = (ImGui::IsWindowHovered()) ? false : true;

		// Change style 
		ImGuiStyle& style = ImGui::GetStyle();
		style.FrameRounding = 5.f;
		style.GrabRounding = 3.f;
		style.WindowRounding = 7.f;

		if (ImGui::Button("Impulse")) UI.impulse_window = true;
		if (UI.impulse_window) {
			RenderImpulseWindow();
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
		if (ImGui::RadioButton("l0", &UI.scalar_type, (int)ScalarType::EVal0)) {
			UpdateScalarField();
		}
		ImGui::SameLine();
		if (ImGui::RadioButton("evec0", &UI.scalar_type, (int)ScalarType::EVec0)) {
			UpdateColormap();
		}

		if (ImGui::RadioButton("l1", &UI.scalar_type, (int)ScalarType::EVal1)) {
			UpdateScalarField();
		}
		ImGui::SameLine();
		if (ImGui::RadioButton("evec1", &UI.scalar_type, (int)ScalarType::EVec1)) {
			UpdateColormap();
		}

		if (ImGui::RadioButton("l2", &UI.scalar_type, (int)ScalarType::EVal2)) {
			UpdateScalarField();
		}
		ImGui::SameLine();
		if (ImGui::RadioButton("evec2", &UI.scalar_type, (int)ScalarType::EVec2)) {
			UpdateColormap();
		}


		/*
		if (ImGui::BeginTabBar("MyTabBar"))
		{

			// 1st tab
			if (ImGui::BeginTabItem("Tensor Field"))
			{
				////////////////////////////////////////////////  Load tensor field  ///////////////////////////////////////////////
				ImGui::Dummy(ImVec2(0.0f, 7.5f));
				ImGui::SeparatorText("Load");
				ImGui::Dummy(ImVec2(0.0f, 2.5f));
				if (ImGui::Button("Load Tensor"))					                                // create a button for loading the shader
				{
					if (ImGui::Button("Load Field"))					                        // create a button for loading the field
						ImGuiFileDialog::Instance()->OpenDialog("LoadNpyFile", "Choose NPY File", ".npy,.npz");
					ImGui::OpenPopup("Open File");
					UI.field_loaded = true;
				}
				if (ImGuiFileDialog::Instance()->Display("LoadNpyFile")) {				    // if the user opened a file dialog
					if (ImGuiFileDialog::Instance()->IsOk()) {								// and clicks okay, they've probably selected a file
						const std::string filename = ImGuiFileDialog::Instance()->GetFilePathName();	// get the name of the file

						if (const std::string extension = filename.substr(filename.find_last_of('.') + 1); extension == "npy") {
							UI.loaded_filename = filename;                                  // store the file name in the UI
							LoadTensorField(UI.loaded_filename, &T0);
							UI.field_loaded = true;
							Tn = T0;
							EigenDecomposition(&Tn, &Lambda, &ThetaPhi, UI.cuda_device);
							RefreshScalarField();
							RefreshVisualization();
						}
					}
					ImGuiFileDialog::Instance()->Close();									// close the file dialog box
				}
				//if (file_dialog.showFileDialog("Open File", imgui_addons::ImGuiFileBrowser::DialogMode::OPEN, ImVec2(700, 700), ".npy"))
				//	OpenFileDialog();

				ImGui::Dummy(ImVec2(0.0f, 7.5f));
				if (ImGui::InputFloat3("Voxel Sizes", &in_voxelsize[0])) {
					if (in_voxelsize[0] < epsilon) in_voxelsize[0] = epsilon;
					if (in_voxelsize[1] < epsilon) in_voxelsize[1] = epsilon;
					if (in_voxelsize[2] < epsilon) in_voxelsize[2] = epsilon;
					T0.spacing(in_voxelsize[0], in_voxelsize[1], in_voxelsize[2]);
					Tn.spacing(in_voxelsize[0], in_voxelsize[1], in_voxelsize[2]);
					UpdateCamera();
				}

				ImGui::Dummy(ImVec2(0.0f, 7.5f));
				ImGui::PushItemWidth(-100);
				ImGui::SliderFloat("alpha", &alpha, 0.01f, 1.0f, "%.2f");
				ImGui::Dummy(ImVec2(0.0f, 2.5f));

				///////////////////////////////////////////////  Render Planes  //////////////////////////////////////////////////

				ImGui::SeparatorText("Planes");
				ImGui::Dummy(ImVec2(0.0f, 2.5f));

				ImGui::Checkbox("X", &RENDER_PLANE[0]);
				ImGui::SetItemTooltip("Render plane X of the volume.");
				ImGui::SameLine();
				ImGui::SliderInt("##XPosition", &PLANE_POSITION[0], 0, Tn.X() - 1);

				ImGui::Checkbox("Y", &RENDER_PLANE[1]);
				ImGui::SetItemTooltip("Render plane Y of the volume.");
				ImGui::SameLine();
				ImGui::SliderInt("##YPosition", &PLANE_POSITION[1], 0, Tn.Y() - 1);

				ImGui::Checkbox("Z", &RENDER_PLANE[2]);
				ImGui::SetItemTooltip("Render plane Z of the volume.");
				ImGui::SameLine();
				ImGui::SliderInt("##ZPosition", &PLANE_POSITION[2], 0, Tn.Z() - 1);

				ImGui::Checkbox("F", &RENDER_EN_FACE);
				ImGui::SetItemTooltip("Render a camera-oriented en face plane.");
				ImGui::SameLine();
				ImGui::SliderFloat("##EFPosition", &EN_FACE_POSITION, -Tn.smax(), Tn.smax());

				///////////////////////////////////////////////  Render Volume  //////////////////////////////////////////////////

				ImGui::SeparatorText("Volume");
				ImGui::Dummy(ImVec2(0.0f, 2.5f));

				ImGui::Checkbox("V", &RENDER_VOLUMETRIC);
				ImGui::SetItemTooltip("Render a camera-oriented en face plane.");
				ImGui::SameLine();
				ImGui::SliderInt("##VPlanes", &VOLUMETRIC_PLANES, 0, 4 * std::max(Tn.X(), std::max(Tn.Y(), Tn.Z())));
				ImGui::Dummy(ImVec2(0.0, 5.0f));

				///////////////////////////////////////////////   Cropping   ///////////////////////////////////////////////////

				ImGui::Dummy(ImVec2(0.0, 5.0f));
				ImGui::LabelText("Width", "Position");

				float child_width = ImGui::GetWindowWidth() - ImGui::GetStyle().WindowPadding.x * 4 - ImGui::GetStyle().ItemInnerSpacing.x * 2;

				// position control
				ImGui::BeginChild("position", ImVec2(child_width / 2.0f, 175), ImGuiChildFlags_Border);
				for (int i = 0; i < 3; i++) {
					if (i > 0) ImGui::SameLine();
					ImGui::PushID(i);
					ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor::HSV(i * 2 / 7.0f, 0.5f, 0.5f));
					ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, (ImVec4)ImColor::HSV(i * 2 / 7.0f, 0.6f, 0.5f));
					ImGui::PushStyleColor(ImGuiCol_FrameBgActive, (ImVec4)ImColor::HSV(i * 2 / 7.0f, 0.7f, 0.5f));
					ImGui::PushStyleColor(ImGuiCol_SliderGrab, (ImVec4)ImColor::HSV(i * 2 / 7.0f, 0.9f, 0.9f));
					int size = (i == 0) ? Tn.sx() : ((i == 1) ? Tn.sy() : Tn.sz());
					ImGui::VSliderInt("##pos", ImVec2(50, 150), &position_values[i], 1, size);
					ImGui::PopStyleColor(4);
					ImGui::PopID();
				}
				ImGui::EndChild();
				ImGui::SameLine();

				// width control
				ImGui::BeginChild("width", ImVec2(child_width / 2.0f, 175), ImGuiChildFlags_Border);
				for (int i = 0; i < 3; i++) {
					ImGui::PushStyleVar(ImGuiStyleVar_SelectableTextAlign, ImVec2(child_width / 2.0f, child_width / 2.0f));
					if (i > 0) ImGui::SameLine();
					ImGui::PushID(i);
					ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor::HSV(i * 2 / 7.0f, 0.5f, 0.5f));
					ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, (ImVec4)ImColor::HSV(i * 2 / 7.0f, 0.6f, 0.5f));
					ImGui::PushStyleColor(ImGuiCol_FrameBgActive, (ImVec4)ImColor::HSV(i * 2 / 7.0f, 0.7f, 0.5f));
					ImGui::PushStyleColor(ImGuiCol_SliderGrab, (ImVec4)ImColor::HSV(i * 2 / 7.0f, 0.9f, 0.9f));
					int width_max = (i == 0) ? Tn.sx() : ((i == 1) ? Tn.sy() : Tn.sz());
					ImGui::VSliderInt("##width", ImVec2(50, 150), &width_values[i], 0, width_max);
					ImGui::PopStyleColor(4);
					ImGui::PopStyleVar();
					ImGui::PopID();
				}
				ImGui::EndChild();
				ImGui::Dummy(ImVec2(0.0f, 7.5f));

				////////////////////////////////////////////  Scalar Visualization  /////////////////////////////////////////////

				ImGui::SeparatorText("Scalar Visualization");
				ImGui::Dummy(ImVec2(0.0f, 2.5f));
				bool eigenvalues = ImGui::RadioButton("Eigenvalues", &SCALAR_TYPE, ScalarType::EVal);
				if (eigenvalues) {
					ColormapEval(SCALAR_EVAL);
				}

				if (ImGui::InputInt("Eigenvalue", &SCALAR_EVAL)) {
					if (SCALAR_EVAL < 0) SCALAR_EVAL = 0;
					if (SCALAR_EVAL > 2) SCALAR_EVAL = 2;
					if (SCALAR_TYPE == ScalarType::EVal) ColormapEval(SCALAR_EVAL);
				}
				if (ImGui::RadioButton("Anisotropy", &SCALAR_TYPE, ScalarType::Anisotropy)) {
					ColormapScalar();
				}
				const char* items[] = { "Fractional", "Linear", "Plate", "Spherical" };

				// Pass in the preview value visible before opening the combo (it could technically be different contents or not pulled from items[])
				const char* combo_preview_value = items[SCALAR_ANISOTROPY];

				if (ImGui::BeginCombo("##AnisotropyType", combo_preview_value))
				{
					for (int n = 0; n < IM_ARRAYSIZE(items); n++)
					{
						const bool is_selected = (SCALAR_ANISOTROPY == n);
						if (ImGui::Selectable(items[n], is_selected)) {
							SCALAR_ANISOTROPY = n;
							ColormapScalar();
						}

						// Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
						if (is_selected)
							ImGui::SetItemDefaultFocus();
					}

					ImGui::EndCombo();
				}

				if (ImGui::RadioButton("Eigenvectors", &SCALAR_TYPE, ScalarType::EVec)) {
					ColormapEvec(2);
				}
				if (ImGui::InputInt("Eigenvector", &SCALAR_EVEC)) {
					if (SCALAR_EVEC < 0) SCALAR_EVEC = 0;
					if (SCALAR_EVEC > 2) SCALAR_EVEC = 2;
					if (SCALAR_TYPE == ScalarType::EVec) ColormapEvec(SCALAR_EVEC);
				}
				//ImGui::PopItemWidth();
				ImGui::Dummy(ImVec2(0.0f, 7.5f));

				//////////////////////////////////////////  Orthogonal / Perspective View  /////////////////////////////////////////

				// Use perspective view instead of ortho view
				ImGui::SeparatorText("Projection");
				ImGui::Dummy(ImVec2(0.0f, 2.5f));

				if (ImGui::RadioButton("Ortho", !perspective))
					perspective = false;
				ImGui::SameLine();
				if (ImGui::RadioButton("Perspective", perspective))
					perspective = true;
				ImGui::Spacing(); ImGui::Spacing();

				ImGui::Dummy(ImVec2(0.0f, 7.5f));

				//////////////////////////////////////////////////     Zoom      //////////////////////////////////////////////////
				// Zooming in and out option
				ImGui::SeparatorText("Zoom");
				ImGui::Dummy(ImVec2(0.0f, 2.5f));

				ImGui::InputFloat("##Zoom", &zoom, 0.1f, 2.0f);
				zoom = (zoom < 1.0f) ? 1.0f : ((zoom > 5) ? 5 : zoom);
				ImGui::SameLine();
				if (ImGui::Button("O", ImVec2(25, 25))) zoom = 1.0f;             // reset zoom
				ImGui::Spacing(); ImGui::Spacing();
				ImGui::Dummy(ImVec2(0.0f, 5.0f));

				///////////////////////////////////////////////////   Reset View   //////////////////////////////////////////////////
				ImGui::SeparatorText("Reset");
				ImGui::Dummy(ImVec2(0.0f, 2.5f));
				//float avail = ImGui::GetContentRegionAvail().x;
				//float off = (avail - 50) * 0.5f;
				//if (off > 0.0f)
				//	ImGui::SetCursorPosX(ImGui::GetCursorPosX() + off);
				RESET = ImGui::Button("Reset", ImVec2(50, 25));

				ImGui::EndTabItem();
			}

			// 2nd tab
			if (ImGui::BeginTabItem("Processing"))
			{
				ImGui::Dummy(ImVec2(0.0f, 7.5f));
				ImGui::SeparatorText("Filter");
				ImGui::Dummy(ImVec2(0.0, 5.0f));

				if (ImGui::RadioButton("None", &PROCESSINGTYPE, (int)ProcessingType::NoProcessing) ||
					(PROCESSINGTYPE == ProcessingType::Vote && !TV_STICK && !TV_PLATE)) {
					ResetField();
					UpdateEigens();
					ScalarRefresh();
				}
				ImGui::Dummy(ImVec2(0.0f, 5.0f));
				///////////////////////////////////////////////  Gaussian Blur  ///////////////////////////////////////////////////

				if (ImGui::RadioButton("Gaussian Blur", &PROCESSINGTYPE, (int)ProcessingType::Gaussian)) {
					if (PROCESSINGTYPE == ProcessingType::Gaussian) {
						GaussianFilter(SIGMA);
						UpdateEigens();
						ScalarRefresh();
					}
				}
				ImGui::SameLine();
				if (ImGui::InputFloat("##Sigma", &SIGMA, 0.2f, 1.0f)) {
					if (SIGMA < 0.0f) SIGMA = 0.0f;
					if (PROCESSINGTYPE == ProcessingType::Gaussian) {
						GaussianFilter(SIGMA);
						UpdateEigens();
						ScalarRefresh();
					}
				}

				///////////////////////////////////////////////  Tensor Voting  ///////////////////////////////////////////////////

				if (ImGui::RadioButton("Tensor Voting", &PROCESSINGTYPE, (int)ProcessingType::Vote)) {
					UpdateTensorField();
				}

				if (ImGui::InputFloat("Sigma 1", &TV_SIGMA1, 0.2f, 1.0f) && PROCESSINGTYPE == ProcessingType::Vote) {
					TV_SIGMA1 = std::max(0.0f, TV_SIGMA1);
					if (ImGui::IsItemEdited()) UpdateTensorField();
				}

				if (ImGui::InputFloat("Sigma 2", &TV_SIGMA2, 0.2f, 1.0f) && PROCESSINGTYPE == ProcessingType::Vote) {
					TV_SIGMA2 = std::max(0.0f, TV_SIGMA2);
					if (ImGui::IsItemEdited()) UpdateTensorField();
				}

				if (ImGui::InputInt("Power", &TV_POWER, 1, 5) && PROCESSINGTYPE == ProcessingType::Vote) {
					TV_POWER = std::max(1, TV_POWER);
					UpdateTensorField();
				}

				if (ImGui::Checkbox("Stick", &TV_STICK) && PROCESSINGTYPE == ProcessingType::Vote)
					UpdateTensorField();
				ImGui::SameLine();
				if (ImGui::Checkbox("Plate", &TV_PLATE) && PROCESSINGTYPE == ProcessingType::Vote)
					UpdateTensorField();

				ImGui::EndTabItem();
			}

			// 3rd tab
			if (ImGui::BeginTabItem("Data"))
			{
				ImGui::Dummy(ImVec2(0.0f, 7.5f));
				ImGui::SeparatorText("Eigenvalues");
				ImGui::Dummy(ImVec2(0.0, 5.0f));
				static ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;
				if (ImGui::BeginTable("table_eigenvalues", 3, flags))
				{
					ImGui::TableSetupColumn("Eigenvalues");
					ImGui::TableSetupColumn("min");
					ImGui::TableSetupColumn("max");
					ImGui::TableHeadersRow();

					for (int row = 0; row < 3; row++)
					{
						ImGui::TableNextRow();
						ImGui::TableNextColumn();
						ImGui::Text("L%d", row);
						ImGui::TableNextColumn();
						ImGui::Text("%.3f", Ln.channel(row).minv());
						ImGui::TableNextColumn();
						ImGui::Text("%.3f", Ln.channel(row).maxv());
					}
					ImGui::EndTable();
				}
				// we are only interested in the x,y,z coordinates
				//std::vector<size_t> smallest_loc = Ln.index_of(Ln.channel(0).minv());						// location of the minimum value of the smallest eigenvalue
				//std::vector<size_t> largest_loc = Ln.index_of(Ln.channel(2).maxv());						// location of the maximum value of the largest eigenvalue

				//// corresponding eigenvectors of the minimum of the smallest eigenvalue
				//float theta1 = Vn(smallest_loc[2], smallest_loc[1], smallest_loc[0], 0);
				//float phi1 = Vn(smallest_loc[2], smallest_loc[1], smallest_loc[0], 1);
				//float theta2 = Vn(smallest_loc[2], smallest_loc[1], smallest_loc[0], 2);
				//float phi2 = Vn(smallest_loc[2], smallest_loc[1], smallest_loc[0], 3);
				//std::vector<std::vector<float>> smallest_eigenvector = GetEigenVectors(theta1, theta2, phi1, phi2);

				ImGui::Dummy(ImVec2(0.0f, 7.5f));
				ImGui::SeparatorText("Eigenvectors of L0 (smallest L)");
				ImGui::Dummy(ImVec2(0.0, 5.0f));

				if (ImGui::BeginTable("table_eigenvector_L0", 4, flags))
				{
					const char* row_labels[3] = { "V1", "V2", "V3" };
					for (int row = 0; row < 3; row++)
					{
						ImGui::TableNextRow();
						ImGui::TableNextColumn();
						ImGui::Text("%s", row_labels[row]);

						for (int col = 0; col < 3; col++)
						{
							ImGui::TableNextColumn();
							ImGui::Text("%.3f", Vn.channel(row).minv());
						}
					}

					ImGui::EndTable();
				}

				ImGui::EndTabItem();
			}

			// 4th tab
			if (ImGui::BeginTabItem("Profiling")) {

				///////////////////////////////////////////////  Memory bar  ///////////////////////////////////////////////////
				ImGui::SeparatorText("CUDA Device");
				ImGui::Text("Device ID: %d", in_device);
				ImGui::Dummy(ImVec2(0.0f, 2.5f));
				float free_m, total_m, used_m;
				size_t free_t, total_t;
				char buf[64];
				if (in_device >= 0) {
					cudaMemGetInfo(&free_t, &total_t);
					free_m = static_cast<float>(free_t) / (1048576.0f);
					total_m = static_cast<float>(total_t) / (1048576.0f);
					used_m = total_m - free_m;
				}
				else {								// CPU memory will be added later
					used_m = 0.0f;
					total_m = 0.0f;
				}
				sprintf(buf, "%.1f/%.1f MB", used_m, total_m);
				float bar_size = ImGui::GetWindowWidth() - ImGui::CalcTextSize("Memory Usage").x -
					ImGui::GetStyle().WindowPadding.x * 2 - ImGui::GetStyle().ItemInnerSpacing.x;
				ImGui::ProgressBar((used_m / total_m), ImVec2(bar_size, 0.0f), buf);
				ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
				ImGui::Text("Memory Usage");
				ImGui::EndTabItem();

				ImGui::SeparatorText("Timers");
				ImGui::Text("Load Field: %f s", t_loading);
				ImGui::Text("Reset Field: %f s", t_resetfield);
				ImGui::Text("Eigendecomposition Calculation: %f s", t_eigendecomposition);
				ImGui::Text("Colormap Eigenvalues: %f s", t_cmap_eval);
				ImGui::Text("Colormap Eigenvectors: %f s", t_cmap_evec);
				ImGui::Text("Colormap Fractional Anisotropy: %f s", t_cmap_fa);
				ImGui::Text("Colormap Linear Anisotropy: %f s", t_cmap_linear);
				ImGui::Text("Colormap Plate Anisotropy: %f s", t_cmap_plate);
				ImGui::Text("Colormap Spherical Anisotropy: %f s", t_cmap_sphere);
				ImGui::Text("Gaussian Blur: %f s", t_gaussian);
			}

			ImGui::EndTabBar();
		}

		*/

		ImGui::GetFont()->Scale = old_size;
		ImGui::PopFont();
		ImGui::End();


	}
	//ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);  // Render a separate window showing the FPS

	ImGui::Render();                                                            // Render all windows

}
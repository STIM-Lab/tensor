#include <GL/glew.h>
#include <GLFW/glfw3.h> // Will drag system OpenGL headers

#include "tira/graphics_gl.h"
#include "tira/graphics/glShader.h"
#include "tira/graphics/shapes/circle.h"
#include "tira/image/colormap.h"

#include "tview2.h"






extern TV2_UI UI;

extern tira::image<glm::mat2> T0;
extern tira::image<glm::mat2> Tn;
extern tira::image<float> Scalar;
extern tira::image<float> Theta;
extern tira::image<float> Lambda;

extern GLFWwindow* window;                                     // pointer to the GLFW window that will be created (used in GLFW calls to request properties)
extern const char* glsl_version;              // specify the version of GLSL

//float ui_scale = 1.5f;                                  // scale value for the UI and UI text

const std::string colormap_shader_string =
#include "shaders/colormap.shader"
;

const std::string glyph_shader_string =
#include "shaders/glyph2d.shader"
;

tira::glGeometry cmapGeometry;
tira::glMaterial* cmapMaterial;

tira::glGeometry glyphGeometry;
tira::glMaterial* glyphMaterial;
tira::glMaterial* testmaterial;

/// Generates the glyph geometry and stores it in a geometry structure for rendering
void GenerateGlyphs() {
    //if (!UI.render_glyphs) return;                     // don't update if they aren't being displayed
    if (Tn.size() == 0) return;

    tira::trimesh<float> circle = tira::circle<float>(UI.glyph_tesselation).scale({ 0.0f, 0.0f });

    const tira::trimesh<float> glyph_row = circle.tile({ 1.0f, 0.0f, 0.0f }, Tn.width());
    const tira::trimesh<float> glyph_grid = glyph_row.tile({ 0.0f, 1.0f, 0.0f }, Tn.height());

    glyphGeometry = tira::glGeometry(glyph_grid);

}

/// Generate a colormap from an eigenvector and its associated eigenvalues
tira::image<unsigned char> ColormapEigenvector(const unsigned int i) {

    // calculate a color representing the eigenvector angle
    tira::image<float> AngleColor = Theta.channel(i).cmap(-std::numbers::pi, std::numbers::pi, ColorMap::RainbowCycle);


    // adjust the pixel color based on the eccentricity
    if (UI.eccentricity_color_mode) {
        tira::image<float> EccentricityColor(AngleColor.X(), AngleColor.Y(), AngleColor.C());
        if (UI.eccentricity_color_mode == AdjustColorType::Lighten)
            EccentricityColor = 255;
        else
            EccentricityColor = 0;

        // calculate the eccentricity of the tensor field
        tira::image<float> eccentricity;
        ImageFrom_Eccentricity(&Lambda, &eccentricity);

        // modify the angle color by the eccentricity
        AngleColor = AngleColor * eccentricity + (-eccentricity + 1) * EccentricityColor;
    }

    if (UI.magnitude_color_mode) {
        tira::image<float> L1 = Lambda.channel(1);                  // get the eigenvalues for the largest eigenvector
        if (UI.signed_eigenvalues == -1)                            // if only negative eigenvalues will be displayed
            L1 = L1.clamp(-INFINITY, 0).abs();                  // clamp all positive eigenvalues to zero
        else if (UI.signed_eigenvalues == +1)                       // if only positive eigenvalues will be displayed
            L1 = L1.clamp(0, INFINITY);                         // clamp all negative eigenvalues to zero
        else                                                        // if all eigenvalues will be displayed
            L1 = L1.abs();                                               // store their absolute values
        if (UI.magnitude_color_threshold < 0)
            UI.magnitude_color_threshold = L1.maxv();

        tira::image<float> L1norm = (L1 / UI.magnitude_color_threshold).clamp(0.0f, 1.0f);

        tira::image<float> MagnitudeColor(AngleColor.X(), AngleColor.Y(), AngleColor.C());
        if (UI.magnitude_color_mode == AdjustColorType::Lighten)
            MagnitudeColor = 255;
        else
            MagnitudeColor = 0;

        AngleColor = AngleColor * L1norm + (-L1norm + 1) * MagnitudeColor;
    }
    AngleColor.save("test_anglecolor.png");
    return AngleColor;
}

/// Generates a new colormap based on the desired visualization parameters and assigns that colormap to the CMAP material
void GenerateColormap() {

    tira::image<unsigned char> ColorMapImage;                // create a temporary image to store the color map
    float ScalarMin = Scalar.minv();                // find the minimum value for the scalar field
    float ScalarMax = Scalar.maxv();                // find the maximum value for the scalar field
    float ScalarMaxMag = std::max(std::abs(ScalarMax), std::abs(ScalarMin));

    switch (UI.scalar_type) {
    case ScalarType::EVal0:
    case ScalarType::EVal1:
    case ScalarType::Tensor00:
    case ScalarType::Tensor11:
    case ScalarType::Tensor01:
    case ScalarType::Eccentricity:
    case ScalarType::LinearEccentricity:
        if (ScalarMin >= 0)
            ColorMapImage = Scalar.cmap(0, ScalarMax, ColorMap::Magma);
        else
            ColorMapImage = Scalar.cmap(-ScalarMaxMag, ScalarMaxMag, ColorMap::Brewer);
        break;
    case ScalarType::EVec0:
        ColorMapImage = ColormapEigenvector(0);
        break;
    case ScalarType::EVec1:
        ColorMapImage = ColormapEigenvector(1);
        break;

    default:
        throw std::runtime_error("Invalid scalar type");
    }
    cmapMaterial->SetTexture("mapped_image", ColorMapImage, GL_RGB8, GL_NEAREST);
}

/// Updates the glyph texture maps with the given eigenvalues and angles
void UpdateGlyphTextures(tira::image<float>* lambda, tira::image<float>* theta) {
    glyphMaterial->SetTexture("lambda", *lambda, GL_RG32F, GL_NEAREST);
    glyphMaterial->SetTexture("evecs", *theta, GL_RG32F, GL_NEAREST);
}



// Calculate the viewport width and height in field pixels given the size of the field and window
void FitRectangleToWindow(float field_width_pixels, float field_height_pixels,
    float window_width, float window_height,// float scale,
    float& viewport_width, float& viewport_height) {

    float display_aspect = window_width / window_height;
    float image_aspect = field_width_pixels / field_height_pixels;
    if (image_aspect > display_aspect) {
        viewport_width = field_width_pixels;// / scale;
        viewport_height = field_width_pixels / display_aspect;// / scale;
    }
    else {
        viewport_height = field_height_pixels;// / scale;
        viewport_width = field_height_pixels * display_aspect;// / scale;
    }
}

GLFWwindow* InitWindow(int width, int height) {
    // Setup window
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        throw std::runtime_error("Failed to initialize GLFW");

    // GL 3.0 + GLSL 130
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);



    // Create window with graphics context
    GLFWwindow* window = glfwCreateWindow(width, height, "ImGui GLFW+OpenGL3 Hello World Program", nullptr, nullptr);
    if (window == nullptr)
        throw std::runtime_error("Failed to create GLFW window");

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync
    //if (FIELD_LOADED) {
    glfwSetCursorPosCallback(window, mouse_callback);
    //}
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);

    ImGuiInit(window, glsl_version);
    if (glewInit() != GLEW_OK)
        throw std::runtime_error("Failed to initialize GLEW");

    return window;
}

// initialize everything that will be rendered (colormapped images and glyph geometry)
void InitShaders() {
    cmapMaterial = new tira::glMaterial(colormap_shader_string);
    glyphMaterial = new tira::glMaterial(glyph_shader_string);
}

void InitCmapGeometry() {
    cmapGeometry = tira::glGeometry::GenerateRectangle<float>();
}

/// Refresh the entire visualizaiton, including glyphs, camera parameters, and scalar fields
void RefreshVisualization() {
    GenerateColormap();
    UpdateGlyphTextures(&Lambda, &Theta);
    UI.camera_position = glm::vec3(static_cast<float>(Tn.width()) / 2.0f, static_cast<float>(Tn.height()) / 2.0f, 0.0f);
}

/// Render the tensor field using OpenGL. This includes a color-mapped image and/or glyphs.
void RenderFieldOpenGL(GLint display_w, GLint display_h) {


    // Calculate the viewport width and height based on the dimensions of the tensor field and the screen (so that the field isn't distorted)
    FitRectangleToWindow(static_cast<float>(Tn.width()),
        static_cast<float>(Tn.height()),
        static_cast<float>(display_w),
        static_cast<float>(display_h),
        UI.viewport[0], UI.viewport[1]);

    glm::vec2 viewport(UI.viewport[0], UI.viewport[1]);
    glm::vec2 view_extent = viewport * UI.camera_zoom / 2.0f;
    float left = -view_extent[0] + UI.camera_position[0];
    float right = view_extent[0] + UI.camera_position[0];
    float bottom = view_extent[1] + UI.camera_position[1];
    float top = -view_extent[1] + UI.camera_position[1];
    glm::mat4 Mview = glm::ortho(left, right, bottom, top);   // create a view matrix
    glm::vec3 center(static_cast<float>(Tn.width()) / 2.0f, static_cast<float>(Tn.height()) / 2.0f, 0.0f);

    // if the user is visualizing a scalar component of the tensor field as a color map
    if (UI.scalar_type != ScalarType::NoScalar) {

        glm::vec3 scale(static_cast<float>(Tn.width()), static_cast<float>(Tn.height()), 1.0f);
        glm::mat4 Mscale = glm::scale(glm::mat4(1.0f), scale);                                                      // compose the scale matrix from the width and height of the tensor field
        glm::mat4 Mtrans = glm::translate(glm::mat4(1.0f), center);
        //glm::mat4 Mtrans = glm::mat4(1.0f);                                                                         // there is no translation (the 2D field is centered at the origin)

        glm::mat4 Mobj = Mtrans * Mscale;                                                                      // create the transformation matrix
        cmapMaterial->Begin();                                                                                     // begin using the scalar colormap material
        cmapMaterial->SetUniformMat4f("Mview", Mview);                                                                 // pass the transformation matrix as a uniform
        cmapMaterial->SetUniformMat4f("Mobj", Mobj);
        cmapGeometry.Draw();                                                                                       // draw the rectangle
        cmapMaterial->End();                                                                                       // stop using the material
    }
    // if the user is rendering glyphs
    if (UI.render_glyphs) {
        glm::mat4 Mtrans = glm::translate(glm::mat4(1.0f), glm::vec3(0.5f, 0.5f, 0.0f));
        glm::mat4 Mscale = glm::scale(glm::mat4(1.0f), glm::vec3(1.0, 1.0f, 1.0f));
        glm::mat4 Mobj = Mtrans * Mscale;
        glyphMaterial->Begin();
        glyphMaterial->SetUniform1f("scale", UI.glyph_scale);
        if (UI.glyph_normalize)
            glyphMaterial->SetUniform1f("norm", 0.0f);
        else
            glyphMaterial->SetUniform1f("norm", UI.largest_eigenvalue_magnitude);
        glyphMaterial->SetUniformMat4f("Mview", Mview);
        glyphMaterial->SetUniformMat4f("Mobj", Mobj);
        glyphGeometry.Draw();
        glyphMaterial->End();
    }
}



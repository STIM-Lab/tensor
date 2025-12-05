#pragma once

#include <glm/glm.hpp>

#include <tira/volume.h>
#include <tira/obj.h>

#include <tira/shapes.h>
#include <tira/eigen.h>

/**
 * @brief Generate an OBJ file containing colored glyphs representing a 3D tensor field
 * @param Field is a 3D tira::volume of glm::mat3 values representing a tensor field
 * @param obj_filename is the name of the OBJ file that will be created
 * @param sigma is the smallest allowed ratio for l0/l2 and l1/l2 (so that tensors don't get "squished" along one dimension)
 * @param epsilon is a threshold used to cull glyphs that have a small eigenvector (l2 < epsilon will not appear)
 * @param normalize is a boolean value specifying whether or not the glyph sizes are normalized to the largest eigenvalue in the field
 * @param subdivisions is the number of subdivisions used to create the glyph model
 */
static void field2glyphs(tira::volume<glm::mat3>& Field, const std::string obj_filename,
                         const float sigma = 0.05f, const float epsilon = 0.01f,
                         const bool normalize = true, const unsigned int subdivisions = 1) {
    // create an OBJ data structure for a mesh that will contain all glyphs
    tira::obj OBJ;

    // Calculate the largest eigenvalue in the field so that glyphs can be normalized.
    float max_l2 = 0;

    // This calculation will only be done if normalization is specified
    if (normalize) {
        // For each pixel in the field
        for (size_t zi = 0; zi < Field.Z(); zi++) {
            for (size_t yi = 0; yi < Field.Y(); yi++) {
                for (size_t xi = 0; xi < Field.X(); xi++) {

                    // Perform an eigendecomposition to extract the eigenvalues
                    glm::mat3 t = Field(xi, yi, zi);
                    float lambda[3];
                    tira::shared::eval3_symmetric<float>(t[0][0], t[0][1], t[1][1], t[0][2], t[1][2], t[2][2],
                        lambda[0], lambda[1], lambda[2]);

                    // Update the maximum value if the current tensor has a larger eigenvalue
                    if (lambda[2] > max_l2)
                        max_l2 = lambda[2];
                }
            }
        }
    }

    // Iterate throught the field and generate a glyph for each voxel
    for (size_t zi = 0; zi < Field.Z(); zi++) {
        for (size_t yi = 0; yi < Field.Y(); yi++) {
            for (size_t xi = 0; xi < Field.X(); xi++) {

                // Get the current tensor
                glm::mat3 t = Field(xi, yi, zi);
                float lambda[3];

                // Calculate the eigenvalues
                tira::shared::eval3_symmetric<float>(t[0][0], t[0][1], t[1][1], t[0][2], t[1][2], t[2][2],
                    lambda[0], lambda[1], lambda[2]);

                // Calculate the eigenvectors
                float evec0[3];
                float evec1[3];
                float evec2[3];
                tira::shared::evec3_symmetric<float>(t[0][0], t[0][1], t[1][1], t[0][2], t[1][2], t[2][2],
                    (const float*)lambda, evec0, evec1, evec2);

                // Create a new material and assign a color to the diffuse component based on the largest eigenvector direction
                tira::mtl new_material;
                new_material.Kd[0] = std::abs(evec2[0]);
                new_material.Kd[1] = std::abs(evec2[1]);
                new_material.Kd[2] = std::abs(evec2[2]);

                // if the largest eigenvalue is less than epsilon, skip this glyph
                if (lambda[2] < epsilon) continue;
                if (lambda[1] / lambda[2] < sigma) lambda[1] = sigma * lambda[2];
                if (lambda[0] / lambda[2] < sigma) lambda[0] = sigma * lambda[2];

                // Generate a superquadric glyph, and normalize to the largest eigenvalue (if requested by the user)
                tira::tmesh sq;
                if(normalize)
                    sq = tira::superquadric(lambda[0]/max_l2, lambda[1]/max_l2, lambda[2]/max_l2, 0.5f, 3.0f, subdivisions);
                else
                    sq = tira::superquadric(lambda[0], lambda[1], lambda[2], 0.5f, 3.0f, subdivisions);

                // Generate a rotation matrix to place the glyph in the correct orientation
                glm::mat3 R;
                R[0][0] = evec2[0];
                R[1][0] = evec2[1];
                R[2][0] = evec2[2];
                R[0][1] = evec1[0];
                R[1][1] = evec1[1];
                R[2][1] = evec1[2];
                R[0][2] = evec0[0];
                R[1][2] = evec0[1];
                R[2][2] = evec0[2];

                tira::tmesh sq_rotate = sq.Transform(glm::transpose(R), 0);
                //tira::tmesh sq_rotate = sq.Transform(R, 0);
                tira::tmesh sq_translate = sq_rotate.Translate({ (float)xi - Field.X()/2.0f, (float)yi - Field.Y() / 2.0f, (float)zi - Field.Z() / 2.0f }, 0);

                // Output the material name and add the glyph to a mesh
                std::stringstream ss;
                ss << xi << "_" << yi << "_" << zi;
                new_material.name = ss.str();
                OBJ.AddMesh(sq_translate, ss.str(), new_material);
            }
        }
    }

    OBJ.Save(obj_filename);
}
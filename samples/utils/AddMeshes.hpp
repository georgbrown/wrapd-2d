// Copyright (c) 2021 George E. Brown and Rahul Narain
//
// WRAPD uses the MIT License (https://opensource.org/licenses/MIT)
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is furnished
// to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
// PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// Originally by Matt Overby (http://www.mattoverby.net)
// Modified by George E. Brown (https://www-users.cse.umn.edu/~brow2327/)

//
// Classes/functions used by wrapd samples
//

#ifndef SAMPLES_UTILS_ADDMESHES_HPP_
#define SAMPLES_UTILS_ADDMESHES_HPP_

#include <memory>
#include <vector>

#include "MeshIO.hpp"
#include "Solver.hpp"

// Glue code to couple admm-elastic with mclscene data types
namespace binding {

    static inline void add_trimesh(
            wrapd::Solver *solver,
            std::shared_ptr<mcl::TriangleMesh> &rest_mesh,
            std::shared_ptr<mcl::TriangleMesh> &init_mesh,
            const std::vector<int> &pins,
            const wrapd::Settings &settings,
            bool verbose = true);

}  // namespace binding

//
//  Implementation
//

static inline void binding::add_trimesh(
        wrapd::Solver *solver,
        std::shared_ptr<mcl::TriangleMesh> &rest_mesh,
        std::shared_ptr<mcl::TriangleMesh> &init_mesh,
        const std::vector<int> &pins,
        const wrapd::Settings& settings,
        bool verbose) {
    if (rest_mesh->vertices.size() != init_mesh->vertices.size()) {
        throw std::runtime_error("Error: Rest mesh and init mesh do not have the same number of vertices.");
    }
    // Add vertices to the solver
    int num_tri_verts = rest_mesh->vertices.size();  // tri verts
    int prev_tri_verts = solver->m_system->num_all_verts();
    int num_tris = rest_mesh->faces.size();

    // Add nodes to the solver
    wrapd::math::MatX2 added_X(num_tri_verts, 2);
    wrapd::math::MatX2 added_X_init(num_tri_verts, 2);
    for (int i = 0; i < num_tri_verts; ++i) {
        int idx = i + prev_tri_verts;
        added_X.row(idx) = (rest_mesh->vertices[i]).segment<2>(0);
        added_X_init.row(idx) = (init_mesh->vertices[i]).segment<2>(0);
    }
    solver->m_system->add_nodes(added_X, added_X_init);

    std::vector<int> sorted_pins(pins);
    std::sort(sorted_pins.begin(), sorted_pins.end());
    std::vector<int> free_index_list(rest_mesh->vertices.size());
    std::vector<int> fix_index_list(rest_mesh->vertices.size());
    
    for (int i = 0; i < static_cast<int>(fix_index_list.size()); i++) {
        fix_index_list[i] = -1;
    }

    int j = 0;
    int k = 0;
    for (int i = 0; i < static_cast<int>(free_index_list.size()); i++) {
        if (j < static_cast<int>(sorted_pins.size()) && i == sorted_pins[j]) {
            free_index_list[i] = -1;
            fix_index_list[i] = j;
            j += 1;
        } else {
            free_index_list[i] = k;
            fix_index_list[i] = -1;
            k += 1;
        }
    }

    std::shared_ptr<wrapd::TriElements>& tri_elements = solver->m_system->tri_elements();
    wrapd::create_tris_from_mesh<double, wrapd::TriElements>(
            tri_elements,
            &rest_mesh->vertices[0][0],
            &rest_mesh->faces[0][0],
            num_tris,
            settings,
            free_index_list,
            fix_index_list,
            prev_tri_verts);

    if (verbose) {
        std::cout << "Input mesh: " <<
            "\n\tvertices: " << num_tri_verts <<
            "\n\ttris: " << num_tris <<
        std::endl;
    }
}

#endif  // SAMPLES_UTILS_ADDMESHES_HPP_

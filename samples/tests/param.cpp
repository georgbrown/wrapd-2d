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
// By George E. Brown (https://www-users.cse.umn.edu/~brow2327/)

#include "MeshIO.hpp"
#include "AddMeshes.hpp"
#include "Settings.hpp"
#include "Math.hpp"
#include "Application.hpp"
#include "Solver.hpp"
#include "AlgorithmData.hpp"
#include "DataLog.hpp"
#include "InitParam.hpp"

#include <algorithm>

DataLog objectives_log("objectives");
DataLog reweighted_log("reweighted");
DataLog inner_count_log("inner_count");
DataLog flips_count_log("flips_count");
DataLog accumulated_time_s_log("accumulated_time_s");

int convergence_strong_index;
wrapd::math::VecX objectives;
wrapd::math::VecXi reweighted;
wrapd::math::VecXi inner_count;
wrapd::math::VecXi flips_count;
wrapd::math::VecX accumulated_time_s;

mcl::TriangleMesh::Ptr create_mesh(const Eigen::MatrixXd &verts, const Eigen::MatrixXi &faces, bool is_flat);

void post_solve(std::shared_ptr<wrapd::Solver> solver);

Eigen::MatrixXd V_input;
Eigen::MatrixXd V_initial_guess;
Eigen::MatrixXd V_uv;
Eigen::MatrixXi F;

int main(int argc, char *argv[]) {
    wrapd::Settings settings;
    settings.parse_args(argc, argv);

    std::stringstream input_file;
    input_file << settings.m_io.input_mesh();

    using namespace std;

    param_utils::dirichlet_on_circle(input_file.str(), V_input, V_initial_guess, F);
    if (param_utils::count_flips(V_input, F, V_initial_guess) > 0) {
        param_utils::tutte_on_circle(input_file.str(), V_input, V_initial_guess, F);
    }

    V_uv = V_initial_guess;

    mcl::TriangleMesh::Ptr input_mesh = create_mesh(V_input, F, false);
    mcl::TriangleMesh::Ptr flattened_mesh = create_mesh(V_uv, F, true);

    // Identify the vertex closest to the center of the initial embedding.
    // Create a pin constraint to hold that vertex in place during the solve.
    std::vector<int> pins;
    wrapd::math::AlignedBox aabb = flattened_mesh->bounds();
    const int num_v = flattened_mesh->vertices.size();
    wrapd::math::Vec3 aabb_center = 0.5 * (aabb.min() + aabb.max());
    int cand_pin_idx = -1;
    double min_dist_sq = 1.e8;
    for (int i = 0; i < num_v; i++) {
        double dist_sq = (flattened_mesh->vertices[i] - aabb_center).squaredNorm();
        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            cand_pin_idx = i;
        }
    }
    pins.clear();
    pins.emplace_back(cand_pin_idx);

    // Create, setup, and run the application
    Application app(settings);
    app.initialize(input_mesh, flattened_mesh, pins);
    bool success = app.run();

    if (success) {
        post_solve(app.solver());

        if (settings.m_io.verbose() > 0) {
            std::cout << "Saving .obj for final parameterization (wrapd-2d/output/param_out.obj)\n";
        }
        wrapd::math::MatX2 final_uv = app.solver()->m_system->X();
        input_mesh->texcoords.resize(input_mesh->vertices.size());
        for (int vt = 0; vt < final_uv.rows(); vt++) {
            input_mesh->texcoords[vt] = final_uv.row(vt);
        }
        std::stringstream param_out_filename;
        param_out_filename << WRAPD_OUTPUT_DIR << "/" << "param_out.obj";
        mcl::meshio::save_obj(input_mesh.get(), param_out_filename.str(), false, true);

        // Write data
        std::stringstream outdir_ss;
        outdir_ss << WRAPD_OUTPUT_DIR << "/";
        if (settings.m_io.should_save_data()) {
            int num_records = std::min(convergence_strong_index+1, static_cast<int>(objectives.size()));
            for (int i = 0; i < num_records; i++) {
                objectives_log.addPoint(i, objectives[i]);
            }
            for (int i = 0; i < num_records; i++) {
                reweighted_log.addPoint(i, reweighted[i]);
            }
            for (int i = 0; i < num_records; i++) {
                flips_count_log.addPoint(i, flips_count[i]);
            }
            inner_count_log.addPoint(0, 0);
            for (int i = 1; i < num_records; i++) {
                inner_count_log.addPoint(i, inner_count[i-1]);
            }
            for (int i = 0; i < num_records; i++) {
                accumulated_time_s_log.addPoint(i, accumulated_time_s[i]);
            }    
            objectives_log.write(outdir_ss.str());
            reweighted_log.write(outdir_ss.str());
            flips_count_log.write(outdir_ss.str());
            inner_count_log.write(outdir_ss.str());
            accumulated_time_s_log.write(outdir_ss.str());
        }
    } 
}

mcl::TriangleMesh::Ptr create_mesh(const Eigen::MatrixXd &verts, const Eigen::MatrixXi &faces, bool is_flat) {
    mcl::TriangleMesh::Ptr mesh = mcl::TriangleMesh::create();
    for (int i = 0; i < verts.rows(); i++) {
        double x = verts(i, 0);
        double y = verts(i, 1);
        double z = is_flat ? 0 : verts(i, 2);
        mesh->vertices.emplace_back(wrapd::math::Vec3(x, y, z));
    }
    for (int i = 0; i < faces.rows(); i++) {
        Eigen::Vector3i face = faces.row(i);
        mesh->faces.emplace_back(face);
    }
    return mesh;
}

void post_solve(std::shared_ptr<wrapd::Solver> solver) {

    wrapd::AlgorithmData* algorithm_data = solver->algorithm_data();

    convergence_strong_index = algorithm_data->convergence_strong_index();
    objectives = algorithm_data->per_iter_objectives();
    reweighted = algorithm_data->per_iter_reweighted();
    flips_count = algorithm_data->per_iter_flips_count();
    inner_count = algorithm_data->per_iter_inner_iters();
    accumulated_time_s = algorithm_data->per_iter_accumulated_time_s();

    if (solver->m_settings.m_io.verbose() > 0) {
        std::cout << std::fixed;
        std::cout << "Local Step (s): " << algorithm_data->m_runtime.local_ms / 1000. << std::endl;
        std::cout << "Global Step (s): " << algorithm_data->m_runtime.global_ms / 1000. << std::endl;
        std::cout << "Refactor (s): " << algorithm_data->m_runtime.refactor_ms / 1000. << std::endl;
        std::cout << "Total reweights: " << reweighted.sum() << std::endl;
    }

    const wrapd::math::VecX& dual_objectives = algorithm_data->per_iter_dual_objectives();

    for (int i = 0; i < objectives.size(); i++) {
        if (flips_count[i] > 0) {
            objectives[i] = dual_objectives[i];
        }
    }

}
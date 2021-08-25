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

#ifndef SAMPLES_UTILS_APPLICATION_HPP_
#define SAMPLES_UTILS_APPLICATION_HPP_

#ifndef IGL_VIEWER_VIEWER_QUIET
#define IGL_VIEWER_VIEWER_QUIET 1
#endif

#include <igl/opengl/glfw/Viewer.h>
#include <igl/boundary_loop.h>

#include "AddMeshes.hpp"
#include "Math.hpp"
#include "Settings.hpp"
#include "Solver.hpp"


class RenderMesh {
 public:

    enum ColorMode {
        UNIFORM,
        PER_FACE_NORMALIZED_WSQ_RATIOS,
        PER_FACE_DISTORTION
    };

    explicit RenderMesh(std::shared_ptr<mcl::TriangleMesh> mesh) {
        m_color_mode = ColorMode::PER_FACE_DISTORTION;
        m_trimesh = mesh;
        m_mesh_id = m_mesh_id_counter;
        m_mesh_id_counter++;
        const int num_verts = m_trimesh->vertices.size();
        const int num_surface_faces = m_trimesh->faces.size();
        m_V.resize(num_verts, 3);
        m_F.resize(num_surface_faces, 3);
        for (int i = 0; i < num_verts; i++) {
            m_V.row(i) = m_trimesh->vertices[i].cast<double>();
        }
        for (int i = 0; i < num_surface_faces; i++) {
            m_F.row(i) = m_trimesh->faces[i];
        }
    }

    const Eigen::Matrix<double, Eigen::Dynamic, 3>& V() const { return m_V; }
    const Eigen::Matrix<int, Eigen::Dynamic, 3>& F() const { return m_F; }

    void V(const Eigen::Matrix<double, Eigen::Dynamic, 3>& V_input) { m_V = V_input; }

    int id() const { return m_mesh_id; }

    ColorMode color_mode() const { return m_color_mode; }

    void color_mode(ColorMode color_mode) {
        m_color_mode = color_mode;
    }

 private:
    std::shared_ptr<mcl::TriangleMesh> m_trimesh;
    Eigen::Matrix<double, Eigen::Dynamic, 3> m_V;
    Eigen::Matrix<int, Eigen::Dynamic, 3> m_F;
    ColorMode m_color_mode;
    int m_mesh_id;
    static int m_mesh_id_counter;
};


class AppController {
 public:
    AppController() :
            m_sim_running(false),
            m_sim_doiter(false),
            m_should_close(false) {}

    bool m_sim_running;  // run sim each frame
    bool m_sim_doiter;  // run a single solver iteration
    bool m_should_close;  // should close the viewer? 

    auto key_callback(igl::opengl::glfw::Viewer &viewer, unsigned char key, int mod)->bool;
    inline void print_help() const;  // Press H for help
};

class Application {
public:
    using Viewer = igl::opengl::glfw::Viewer;
    using ViewerPtr = std::shared_ptr<Viewer>;

    Application(const wrapd::Settings& settings)
        : m_settings(settings),
          m_initialized(false) {
    }

    void initialize(mcl::TriangleMesh::Ptr input_mesh, mcl::TriangleMesh::Ptr flattened_mesh, const std::vector<int>& pins) {
        m_render_mesh = std::make_shared<RenderMesh>(flattened_mesh);
        m_solver = std::make_shared<wrapd::Solver>(m_settings);
        binding::add_trimesh(m_solver.get(), input_mesh, flattened_mesh, pins, m_settings, m_settings.m_io.verbose());
        m_solver->set_pins(pins);
        m_solver->initialize(m_settings);

        if (m_settings.viewer()) {
            m_viewer = std::make_shared<igl::opengl::glfw::Viewer>();
            m_viewer->core().is_animating = 0;
            m_viewer->core().set_rotation_type(igl::opengl::ViewerCore::ROTATION_TYPE_NO_ROTATION);
            m_viewer->core().camera_zoom = 1.5;
            m_controller = std::make_shared<AppController>();
        }

        m_initialized = true;
    }

    bool run() {
        if (!m_initialized) { return false; }

        if (m_settings.viewer()) {
            m_controller->print_help();
            m_solver->setup();
            display();
            m_solver->finish();
        } else {
            m_solver->solve_offline();
        }
        return true;
    }

    std::shared_ptr<wrapd::Solver> solver() { return m_solver; }

private:

    inline bool display();
    inline void render_window_add_mesh(std::shared_ptr<RenderMesh> render_mesh);
    inline void update_viewer(const wrapd::math::MatX2& X);
    inline void compute_coloring(const RenderMesh::ColorMode color_mode, Eigen::MatrixXd& C);
    

    const wrapd::Settings& m_settings;
    std::shared_ptr<wrapd::Solver> m_solver;
    ViewerPtr m_viewer;  // LIBIGL viewer
    std::shared_ptr<AppController> m_controller;
    std::shared_ptr<RenderMesh> m_render_mesh;
    bool m_initialized;

};

//
// Implementation
//

int RenderMesh::m_mesh_id_counter = 0;

// Returns success or failure
inline bool Application::display() {

    if (!m_initialized) { return false; }

    // Add render meshes
    if (m_render_mesh.get() != nullptr) {
        render_window_add_mesh(m_render_mesh);
        // m_dynamic_meshes[i].update(m_solver.get());
    }

    // Game loop
    m_viewer->callback_pre_draw =
    [&](igl::opengl::glfw::Viewer &) {
        //
        // Game loop
        //
        //  Update
        //
        float t = glfwGetTime();
        static float t_old = t;

        if (m_controller->m_sim_running) {
            m_solver->iterate();
            if (t - t_old > 1.0/30.0) {
                wrapd::math::MatX2 X = m_solver->X_current();
                update_viewer(X);
                t_old = t;
            }
            m_controller->m_should_close = m_solver->termination_check();
        } else if (m_controller->m_sim_doiter) {
            m_controller->m_sim_doiter = false;
            m_solver->iterate();
            wrapd::math::MatX2 X = m_solver->X_current();
            update_viewer(X);
            m_controller->m_should_close = m_solver->termination_check();
        }

        return false;
    };


    m_viewer->callback_post_draw =
    [&](igl::opengl::glfw::Viewer &) {
        if (m_controller->m_should_close) {
            glfwSetWindowShouldClose(m_viewer->window, 1);
        }
        return false;
    };


    const auto &key_down =
            [&](igl::opengl::glfw::Viewer &viewer, unsigned char key, int mod)->bool {
        (void)(mod);
        switch (key) {
            case ' ':
                m_controller->m_sim_running = !m_controller->m_sim_running;
                viewer.core().is_animating ^= 1;
                break;
            case 'I':
            case 'i':
                m_controller->m_sim_doiter = !m_controller->m_sim_doiter;
                break;
            case 'H':
            case 'h':
                m_controller->print_help();
                break;
            default:
                return false;
        }
        return true;
    };

    m_viewer->callback_key_down = key_down;

    m_viewer->launch();

    return true;
}


inline void Application::render_window_add_mesh(std::shared_ptr<RenderMesh> surface) {
    const Eigen::Matrix<double, Eigen::Dynamic, 3>& V = surface->V();
    const Eigen::Matrix<int, Eigen::Dynamic, 3>& F = surface->F();

    int last_index = 0;
    m_viewer->data(last_index).set_mesh(V, F);

    const RenderMesh::ColorMode color_mode = surface->color_mode();
    Eigen::MatrixXd C;
    compute_coloring(color_mode, C);
    m_viewer->data(last_index).set_colors(C);
    m_viewer->data(last_index).show_lines = false;
    m_viewer->data(last_index).shininess = 0.;

    Eigen::VectorXi bnd;
    igl::boundary_loop(F, bnd);
    const int num_bnd_verts = bnd.size();
    Eigen::MatrixXd P1(num_bnd_verts, 3);
    Eigen::MatrixXd P2(num_bnd_verts, 3);
    for (int b = 0; b < num_bnd_verts-1; b++) {
        P1.row(b) = V.row(bnd[b]);
        P2.row(b) = V.row(bnd[b+1]);
    }
    P1.row(num_bnd_verts-1) = V.row(bnd[num_bnd_verts-1]);
    P2.row(num_bnd_verts-1) = V.row(bnd[0]);
    m_viewer->data(0).add_edges(P1, P2, Eigen::RowVector3d(0., 0., 0.));
    m_viewer->data(0).line_width = 3;

    m_viewer->append_mesh();
}


inline void Application::update_viewer(const wrapd::math::MatX2& X) {
    const int num_nodes = X.rows();
    Eigen::MatrixXd P(num_nodes, 3);
    P.setZero();
    for (int i = 0; i < 2; i++) {
        P.col(i) = X.col(i);
    }

    m_render_mesh->V(P);
    Eigen::Matrix<double, Eigen::Dynamic, 3> N_verts;
    const wrapd::math::MatX V = m_render_mesh->V();
    const wrapd::math::MatXi F = m_render_mesh->F();
    int mesh_id = m_render_mesh->id();
    igl::per_face_normals(V, F, N_verts);
    m_viewer->data(mesh_id).clear();
    m_viewer->data(mesh_id).set_mesh(V, F);
    m_viewer->data(mesh_id).set_normals(N_verts);
    m_viewer->data(mesh_id).show_lines = false;
    m_viewer->data(mesh_id).shininess = 0.;

    const RenderMesh::ColorMode color_mode = m_render_mesh->color_mode();
    Eigen::MatrixXd C;
    compute_coloring(color_mode, C);
    m_viewer->data(mesh_id).set_colors(C);

    Eigen::VectorXi bnd;
    igl::boundary_loop(F, bnd);
    const int num_bnd_verts = bnd.size();
    Eigen::MatrixXd P1(num_bnd_verts, 3);
    Eigen::MatrixXd P2(num_bnd_verts, 3);
    for (int b = 0; b < num_bnd_verts-1; b++) {
        P1.row(b) = P.row(bnd[b]);
        P2.row(b) = P.row(bnd[b+1]);
    }
    P1.row(num_bnd_verts-1) = P.row(bnd[num_bnd_verts-1]);
    P2.row(num_bnd_verts-1) = P.row(bnd[0]);
    m_viewer->data(0).add_edges(P1, P2, Eigen::RowVector3d(0., 0., 0.));
    m_viewer->data(0).line_width = 3;
}


inline void Application::compute_coloring(const RenderMesh::ColorMode color_mode, Eigen::MatrixXd& C) {
    (void)(color_mode);
    (void)(C);
    if (color_mode == RenderMesh::ColorMode::PER_FACE_DISTORTION) {
        std::vector<double> distortion = m_solver->m_system->get_distortion();

        double minval = 0.;
        double maxval = 1.;
        wrapd::math::VecX per_face_data(distortion.size());
        for (int j = 0; j < per_face_data.size(); j++) {
            per_face_data[j] = wrapd::math::clamp(0., (distortion[j] - minval) / (maxval - minval), 1.0);
        }

        C.resize(per_face_data.size(), 3);
        for (int i = 0; i < C.rows(); i++) {
            C.row(i) = wrapd::math::Vec3(0.55, 0.55, 0.55) - per_face_data[i] * wrapd::math::Vec3(-0.45, 0.55, 0.55);
        }
     
    } else {
        throw std::runtime_error("Error: Invalid color mode.");
    }
}


inline void AppController::print_help() const {
    std::stringstream ss;
    ss << "\n==========================================\nKeys:\n" <<
        "\t esc:      exit app\n" <<
        "\t spacebar: toggle sim (start/pause)\n" <<
        "\t i:        run a single iteration\n" <<
    "==========================================\n";
    printf("%s", ss.str().c_str());
}


#endif  // SAMPLES_UTILS_APPLICATION_HPP_
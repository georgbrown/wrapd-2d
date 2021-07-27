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

#ifndef SRC_GLOBALSTEPBEHAVIOR_HPP_
#define SRC_GLOBALSTEPBEHAVIOR_HPP_

#include "Math.hpp"
#include "Settings.hpp"
#include "System.hpp"
#include "AlgorithmData.hpp"
#include "MCL/SplitLBFGS2D.hpp"

namespace wrapd {

class GlobalStepBehavior {
 public:
    GlobalStepBehavior(
            const Settings &settings,
            std::shared_ptr<System> system,
            AlgorithmData& algorithm_data) 
        : m_settings(settings),
          m_system(system),
          m_algorithm_data(algorithm_data) {

        // The weight and selector matrices are constructed. We loop through all the local
        // energy terms to gather the data.
        initialize_selectors_and_weights();
    }

    virtual void print_timing() {}

    virtual ~GlobalStepBehavior() {}

    virtual void global_step() = 0;

    virtual double last_step_delta_auglag() { return 1.e8; }

    void initialize_selectors_and_weights() {
        mcl::MicroTimer t;

        // 2. Initialize weights
        t.reset();
        initialize_W();
        m_algorithm_data.m_runtime.setup_solve_ms = t.elapsed_ms();

        // 3. Initialize the A matrix
        t.reset();
        math::Triplets triplets_A;
        m_system->get_A(triplets_A);
        m_A.resize(m_algorithm_data.num_free_verts(), m_algorithm_data.num_free_verts());
        m_A.setFromTriplets(triplets_A.begin(), triplets_A.end());
        m_nonzero_row = std::vector<int>(m_A.rows());
        for (int i = 0; i < m_A.rows(); ++i) {
            m_nonzero_row[i] = m_A.innerVector(i).nonZeros();
        }
        m_A.makeCompressed();
        m_algorithm_data.m_runtime.setup_solve_ms = t.elapsed_ms();
    }

    void initialize_W() {
        math::MatX2 X = m_system->X_init();
        m_system->midupdate_WtW(X);
    }

    virtual void initialize_weights() = 0;
    virtual bool update_weights() = 0;

 protected:
    const Settings& m_settings;
    std::shared_ptr<System> m_system;
    AlgorithmData& m_algorithm_data;
    math::SpMat m_A;
    std::vector<int> m_nonzero_row;


 private:

};


class GlobalProblem : public mcl::optlib::SplitProblem2D<double, Eigen::Dynamic> {
 public:
    GlobalProblem(
            const Settings &settings,
            std::shared_ptr<System> system, 
            AlgorithmData &algorithm_data)
        : m_settings(settings),
          m_system(system),
          m_algorithm_data(algorithm_data) {
    } 
    
    virtual bool converged(
            const math::MatX2 &x0,
            const math::MatX2 &x1,
            const math::MatX2 &grad) override {
        (void)(x0);
        (void)(x1);
        (void)(grad);
        throw std::runtime_error("Error: we are not using the 'converged' function in the global step");
        return false;
    }

    virtual double value(const math::MatX2 &x_free) override {
        return m_system->global_obj_value(x_free);
    }

    virtual double gradient(const math::MatX2 &x_free, math::MatX2 &grad) {
        return m_system->global_obj_grad(x_free, grad);
    }

    virtual void gradient(math::MatX2 &grad) {
        grad.setZero();
        m_system->global_obj_grad(grad);
    }

 private:
    const Settings& m_settings;
    std::shared_ptr<System> m_system;
    AlgorithmData& m_algorithm_data;
};



class UnawareGS : public GlobalStepBehavior {
 public:
    UnawareGS(
            const Settings &settings,
            std::shared_ptr<System> system,
            AlgorithmData &algorithm_data) 
        : GlobalStepBehavior(settings, system, algorithm_data) {

        m_b = math::MatX2::Zero(m_algorithm_data.num_free_verts(), 2);

        #ifdef USE_PARDISO
            m_pardiso_solver.compute(m_A);
        #else
            m_linear_solver.compute(m_A);
        #endif
    }

    virtual void initialize_weights() override {
        m_system->midupdate_WtW();
        m_A = math::SpMat(m_algorithm_data.num_free_verts(), m_algorithm_data.num_free_verts());
        m_A.reserve(m_nonzero_row);
        m_system->update_A(m_A);
        m_A.makeCompressed();

        #ifdef USE_PARDISO
            m_pardiso_solver.factorize(m_A);
        #else
            m_linear_solver.factorize(m_A);
        #endif
    }

    virtual bool update_weights() override {

        bool updated = m_system->possibly_update_weights(m_settings.m_gamma);

        if (updated) {
            m_A = math::SpMat(m_algorithm_data.num_free_verts(), m_algorithm_data.num_free_verts());
            m_A.reserve(m_nonzero_row);
            m_system->update_A(m_A);
            m_A.makeCompressed();

            #ifdef USE_PARDISO                
                m_pardiso_solver.factorize(m_A);
            #else
                m_linear_solver.factorize(m_A);
            #endif
        }
        return updated;  
    }

    virtual void global_step() override {
        mcl::MicroTimer t;
        t.reset();
        m_b = m_system->get_b();

        #ifdef USE_PARDISO
            for (int c = 0; c < 2; c++) {
                m_algorithm_data.m_curr_x_free.col(c) = m_pardiso_solver.solve(m_b.col(c));
            }
        #else
            for (int c = 0; c < 2; c++) {
                m_algorithm_data.m_curr_x_free.col(c) = m_linear_solver.solve(m_b.col(c));
            }
        #endif

        m_system->update_defo_cache(m_algorithm_data.m_curr_x_free, false);
        m_algorithm_data.m_runtime.global_ms += t.elapsed_ms();
        m_system->update_defo_cache(m_algorithm_data.m_curr_x_free, true); // outside runtime tracking, because things like potential_energy_x() look at SVD(Di*x) data
        m_algorithm_data.m_per_iter_inner_iters[m_algorithm_data.m_iter] = 1;
        m_algorithm_data.m_runtime.inner_iters += 1;
    }

 private:
    math::MatX2 m_b;
    #ifdef USE_PARDISO
        mcl::optlib::PardisoSolver m_pardiso_solver;
    #else
        mcl::optlib::LinearSolver m_linear_solver;
    #endif
};


class RotAwareGS : public GlobalStepBehavior {
 public:
    RotAwareGS(
            const Settings &settings,
            std::shared_ptr<System> system,
            AlgorithmData &algorithm_data) 
        : GlobalStepBehavior(settings, system, algorithm_data),
          m_problem(GlobalProblem(settings, system, algorithm_data)) {
        m_lbfgs_solver.update_A0(m_A, true);
        m_lbfgs_solver.update_kappa(settings.m_gstep.kappa());
        m_obj_bef = 1.e8;
        m_obj_aft = 0.;

        Settings::GStep::LineSearch ls = settings.m_gstep.linesearch();
        switch (ls) {
            case Settings::GStep::LineSearch::None: {
                m_lbfgs_solver.m_settings.ls_method = mcl::optlib::LSMethod::None;
                break;
            }
            case Settings::GStep::LineSearch::Backtracking: {
                m_lbfgs_solver.m_settings.ls_method = mcl::optlib::LSMethod::Backtracking;
                break;
            }
            default: {
                throw std::runtime_error("Error: Invalid linesearch");
                break;
            }
        }
        m_lbfgs_solver.m_settings.ls_decrease = 1.e-4;
        m_lbfgs_solver.m_settings.ls_max_iters = 100;
        m_lbfgs_solver.m_settings.max_iters = m_settings.m_gstep.max_iters();
        m_lbfgs_solver.m_settings.verbose = 1;
    }

    virtual void initialize_weights() override {
        m_system->midupdate_WtW();
        m_A = math::SpMat(m_algorithm_data.num_free_verts(), m_algorithm_data.num_free_verts());
        m_A.reserve(m_nonzero_row);
        m_system->update_A(m_A);
        m_A.makeCompressed();
        m_lbfgs_solver.update_A0(m_A, false);
    }

    virtual bool update_weights() override {

        mcl::MicroTimer t;

        bool updated = m_system->possibly_update_weights(m_settings.m_gamma);

        if (updated) {

            static bool first_time = true;

            if (first_time) {
                t.reset();
                math::Triplets triplets_A;

                m_system->get_A(triplets_A);
                m_A.resize(m_algorithm_data.num_free_verts(), m_algorithm_data.num_free_verts());
                m_A.setFromTriplets(triplets_A.begin(), triplets_A.end());

                m_nonzero_row = std::vector<int>(m_A.rows());
                for (int i = 0; i < m_A.rows(); ++i) {
                    m_nonzero_row[i] = m_A.innerVector(i).nonZeros();
                }
                m_A.makeCompressed();

                first_time = false;
            } else {
                m_A = math::SpMat(m_algorithm_data.num_free_verts(), m_algorithm_data.num_free_verts());
                m_A.reserve(m_nonzero_row);
                m_system->update_A(m_A);
                m_A.makeCompressed();
            }
            m_lbfgs_solver.update_A0(m_A, false);        
        }
        return updated;  
    }

    virtual void global_step() override {
        m_obj_bef = m_problem.value(m_algorithm_data.m_curr_x_free);
        
        mcl::MicroTimer t;
        t.reset();

        m_lbfgs_solver.update_max_iters(m_settings.m_gstep.max_iters());
        if (m_algorithm_data.m_iter == 0) {
            m_lbfgs_solver.clear_history();
        }

        // if (m_settings.m_gstep.earlyexit_adaptive()) {
            m_lbfgs_solver.update_earlyexit_tol(std::fabs(m_algorithm_data.m_last_localstep_delta_auglag_obj));
        // } else {
        //    m_lbfgs_solver.update_earlyexit_tol(m_settings.m_gstep.earlyexit_tol());
        // }

        int its = m_lbfgs_solver.minimize(m_problem, m_algorithm_data.m_curr_x_free);
        m_algorithm_data.m_runtime.global_ms += t.elapsed_ms();
        m_algorithm_data.m_per_iter_inner_iters[m_algorithm_data.m_iter] = its;
        m_algorithm_data.m_runtime.inner_iters += its;
        m_obj_aft = m_problem.value(m_algorithm_data.m_curr_x_free);


    }

    virtual double last_step_delta_auglag() { return m_obj_aft - m_obj_bef; }

 private:
    mcl::optlib::SplitLBFGS2D<double, Eigen::Dynamic> m_lbfgs_solver;
    GlobalProblem m_problem;
    double m_obj_bef;
    double m_obj_aft;
};

}  // namespace wrapd

#endif  // SRC_GLOBALSTEPBEHAVIOR_HPP_
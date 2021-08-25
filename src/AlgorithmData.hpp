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

#ifndef SRC_ALGORITHMDATA_HPP_
#define SRC_ALGORITHMDATA_HPP_

#include <memory>
#include <vector>

#include "Math.hpp"
#include "Settings.hpp"
#include "System.hpp"
#include "MCL/MicroTimer.hpp"

namespace wrapd {

// RuntimeData struct used for logging.
// Add timings are per time step.
struct RuntimeData {
    double setup_solve_ms;
    double local_ms;  // total ms for local solver
    double global_ms;  // total ms for global solver
    double refactor_ms;
    int inner_iters;  // total global step iterations
    double inner_error;  // sum of all global step errors (divide by #inner iters to report avg)
    double finish_solve_ms;
    RuntimeData() :
            setup_solve_ms(0),
            local_ms(0),
            global_ms(0),
            refactor_ms(0),
            inner_iters(0),
            inner_error(0),
            finish_solve_ms(0) {}
    void print(const Settings &settings);
};

class AlgorithmData {
 public:
    AlgorithmData();

    void resize_and_zero_residuals(int num_admm_iters);

    void log_initial_data(const Settings& settings, std::shared_ptr<System> system);

    math::VecX per_iter_objectives() const {
        return m_objectives;
    }

    math::VecXi per_iter_reweighted() const {
        return m_reweighted;
    }

    math::VecX per_iter_accumulated_time_s() const {
        return m_accumulated_time_s;
    }

    math::VecXi per_iter_inner_iters() const {
        return m_per_iter_inner_iters;
    }

    math::VecX per_iter_dual_objectives() const {
        return m_dual_objectives;
    }

    math::VecXi per_iter_flips_count() const {
        return m_flips_count;
    }

    void update_per_iter_reweighted(bool reweighted) {
        if (reweighted) {
            m_reweighted[m_iter+1] = 1;
        } else {
            m_reweighted[m_iter+1] = 0;
        }
    }

    void update_per_iter_delta_auglag(double delta_ls, double delta_gs) {
        m_per_iter_ls_delta_auglag[m_iter] = std::fabs(delta_ls);
        m_per_iter_gs_delta_auglag[m_iter] = std::fabs(delta_gs);
    }

    virtual void update_per_iter_residuals(
            const Settings& settings,
            std::shared_ptr<System> system);

    virtual void update_per_iter_runtime();

    virtual void print_initial_data() const;
    virtual void print_curr_iter_data() const;
    virtual void print_final_iter_data() const;

    virtual void reset_fix_free_S_matrix(std::shared_ptr<System> system);

    virtual inline int num_fix_verts() const { return m_S_fix.cols(); }
    virtual inline int num_free_verts() const { return m_S_free.cols(); }

    bool converged_weak() const { return m_converged_weak; }
    bool converged_strong() const { return m_converged_strong; }
    bool earlyexit_asap() const { return m_earlyexit_asap; }

    int convergence_weak_index() const { return m_convergence_weak_index; }
    int convergence_strong_index() const { return m_convergence_strong_index; }

    RuntimeData m_runtime;  // reset each iteration

    double m_init_objective;
    int m_iter;

    math::VecXi m_per_iter_inner_iters;
    math::VecX m_per_iter_ls_delta_auglag;
    math::VecX m_per_iter_gs_delta_auglag;
    math::VecX m_per_iter_gs_early_delta_obj;

    math::MatX2 m_curr_x_free;
    math::MatX2 m_curr_x_fix;
    math::MatX2 m_init_x_fix;

    math::SpMat m_S_fix;
    math::SpMat m_S_free;
    math::VecXi m_positive_pin;

    double m_init_auglag_obj;  // used by the rotation-aware method
    double m_last_localstep_delta_auglag_obj;  // used by the rotation-aware method

 protected:
    double state_x_residual(const math::VecX& x_prev, const math::VecX& x) const;

    math::VecX m_objectives;  // has dim of #admm iters
    math::VecX m_gradnorms;  // has dim of #admm iters

    math::VecXi m_reweighted; // has dim of #admm iters
    math::VecX m_accumulated_time_s;  // has dim of #admm iters

    math::VecX m_dual_objectives;  // has dim of #admm iters
    math::VecXi m_flips_count;

    bool m_converged_weak;
    bool m_converged_strong;
    bool m_earlyexit_asap;
    int m_convergence_weak_index;
    int m_convergence_strong_index;

    Settings::RotAwareness m_rot_awareness;

};  // end of class AlgorithmData

}  // namespace wrapd

#endif  // SRC_ALGORITHMDATA_HPP_

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

#include "AlgorithmData.hpp"
#include <iomanip>

namespace wrapd {

AlgorithmData::AlgorithmData() {
}

void AlgorithmData::log_initial_data(const Settings& settings, std::shared_ptr<System> system) {
    double prev_objective = system->potential_energy();
    double prev_gradnorm = system->potential_gradnorm();
    double prev_dual_objective = system->dual_objective();
    int prev_flips_count = system->inversion_count();

    m_init_objective = 0.25 * (prev_objective) + 1.0;
    m_objectives[0] = 0.25 * (prev_objective) + 1.0;
    m_gradnorms[0] = 0.25 * prev_gradnorm;
    m_dual_objectives[0] = 0.25 * (prev_dual_objective);
    m_flips_count[0] = prev_flips_count;

    // m_optimizer = settings.m_optimizer;
    m_rot_awareness = settings.m_rot_awareness;

}

void AlgorithmData::update_per_iter_residuals(
        const Settings& settings,
        std::shared_ptr<System> system) {

    double curr_objective = system->potential_energy();
    double curr_gradnorm = system->potential_gradnorm();
    double curr_dual_objective = system->dual_objective();
    int curr_flips_count = system->inversion_count();

    m_objectives[m_iter+1] = 0.25 * (curr_objective) + 1.0;
    m_gradnorms[m_iter+1] = 0.25 * curr_gradnorm;
    m_dual_objectives[m_iter+1] = 0.25 * (curr_dual_objective) + 1.0;
    m_flips_count[m_iter+1] = curr_flips_count;

    if (settings.m_earlyexit_asap && m_flips_count[m_iter+1] == 0) {
        m_earlyexit_asap = true;
        m_convergence_strong_index = m_iter+1;
    }

    if (m_converged_weak && m_flips_count[m_iter+1] > 0) {
        m_converged_weak = false;
        m_convergence_weak_index = m_objectives.size() - 1;
    }

    if (m_converged_strong && m_flips_count[m_iter+1] > 0) {
        m_converged_strong = false;
        m_convergence_strong_index = m_objectives.size() - 1;
    }

    // Convergence check
    bool no_recent_flips = m_flips_count[m_iter+1] == 0 && m_flips_count[m_iter] == 0;
    double relative_error = std::fabs(m_objectives[m_iter+1] - m_objectives[m_iter]) / (m_objectives[m_iter+1]);

    if (!m_converged_weak) {
        m_converged_weak = no_recent_flips && (relative_error < 1.e-6 || curr_gradnorm < 1.e-6);
        if (m_converged_weak) {
            m_convergence_weak_index = m_iter+1;
        }
    }

    if (m_converged_weak) {
        m_converged_strong = no_recent_flips && (relative_error < settings.m_earlyexit_tol || curr_gradnorm < settings.m_earlyexit_tol);
        if (m_converged_strong) {
            m_convergence_strong_index = m_iter+1;
        }        
    }

}


void AlgorithmData::update_per_iter_runtime() {
    double curr_accumulated_time_ms =
            m_runtime.local_ms +
            m_runtime.refactor_ms +
            m_runtime.global_ms;
    m_accumulated_time_s[m_iter+1] = curr_accumulated_time_ms / 1000.;
}


void AlgorithmData::print_initial_data() const {
    std::cout << "\nADMM it |  time  |   current objective   | grad norm | RW? |";
    if (m_rot_awareness == Settings::RotAwareness::ENABLED) {
        std::cout << "LS:DeltaAL|GS:DeltaAL| L-BFGS ";
    }
    std::cout << std::endl;
    std::cout << "   init" << " |"
              << std::fixed
              << std::setprecision(3)
              << std::setw(7)
              << m_accumulated_time_s[0] << " | "
              << std::scientific
              << std::setprecision(15)
              << m_objectives[0] << " | "
              << std::setprecision(2)
              << std::setw(9)
              << m_gradnorms[0] << " | "
              << std::setw(3)
              << m_reweighted[0] << " | ";
    if (m_rot_awareness == Settings::RotAwareness::ENABLED) {
        std::cout << "--------" << " | "
              << "--------" << " | "
              << "-----";
    }
    std::cout << std::endl << std::fixed;
}

void AlgorithmData::print_curr_iter_residuals() const {

    std::cout << std::setw(7)
              << m_iter << " |"
              << std::fixed
              << std::setprecision(3)
              << std::setw(7)
              << m_accumulated_time_s[m_iter+1] << " | "
              << std::scientific
              << std::setprecision(15)
              << m_objectives[m_iter+1] << " | "
              << std::setprecision(2)
              << std::setw(9)
              << m_gradnorms[m_iter+1] << " | "
              << std::setw(3)
              << m_reweighted[m_iter+1] << " | ";
    if (m_rot_awareness == Settings::RotAwareness::ENABLED) {
        std::cout << m_per_iter_ls_delta_auglag[m_iter] << " | "
              << m_per_iter_gs_delta_auglag[m_iter] << " |"
              << std::setw(4)
              << m_per_iter_inner_iters[m_iter];
    }
    std::cout << std::endl << std::fixed;
}


void AlgorithmData::resize_and_zero_residuals(int num_admm_iters) {
    m_objectives = math::VecX::Zero(num_admm_iters+1);
    m_gradnorms = math::VecX::Zero(num_admm_iters+1);

    m_reweighted = math::VecXi::Zero(num_admm_iters+1);
    m_accumulated_time_s = math::VecX::Zero(num_admm_iters+1);
    m_dual_objectives = math::VecX::Zero(num_admm_iters+1);
    m_flips_count = math::VecXi::Constant(num_admm_iters+1,-1);

    m_per_iter_inner_iters = math::VecXi::Zero(num_admm_iters);
    m_per_iter_ls_delta_auglag = math::VecX::Zero(num_admm_iters);
    m_per_iter_gs_delta_auglag = math::VecX::Zero(num_admm_iters);
    m_per_iter_gs_early_delta_obj = math::VecX::Zero(num_admm_iters);

    m_reweighted[0] = 1.0;

    m_convergence_weak_index = num_admm_iters;
    m_converged_weak = false;
    m_convergence_strong_index = num_admm_iters;
    m_converged_strong = false;
    m_earlyexit_asap = false;

}

void AlgorithmData::reset_fix_free_S_matrix(std::shared_ptr<System> system) {
    std::shared_ptr<ConstraintSet> constraints = system->constraint_set();

    const int num_all_verts = system->num_all_verts();
    const int pin_dof = static_cast<int>(constraints->m_pins.size());
    const int free_dof = num_all_verts - pin_dof;

    m_positive_pin.setOnes(static_cast<int>(num_all_verts));

    math::SpMat S_fix(num_all_verts, pin_dof);
    math::SpMat S_free(num_all_verts, free_dof);

    Eigen::VectorXi nnz = Eigen::VectorXi::Ones(num_all_verts);  // non zeros per column
    S_fix.reserve(nnz);
    S_free.reserve(nnz);

    int count = 0;
    std::unordered_map<int, math::Vec2>::iterator pinIter = constraints->m_pins.begin();
    for (; pinIter != constraints->m_pins.end(); ++pinIter) {
        // Construct Selection Matrix to select x
        int pin_id = pinIter->first;
        S_fix.coeffRef(pin_id, count) = 1.0;
        m_positive_pin(pin_id) = 0;

        count++;
    }

    count = 0;
    for (int i = 0; i < m_positive_pin.size(); ++i) {
        if (m_positive_pin(i) > 0) {  // Free point
            S_free.coeffRef(i, count) = 1.0;
            count++;
        }
    }

    m_S_fix = S_fix;
    m_S_free = S_free;
}


}  // namespace wrapd

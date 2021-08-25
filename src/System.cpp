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

#include "System.hpp"

namespace wrapd {

System::System(const Settings &settings)
    : m_settings(settings) {
    m_constraints = std::make_shared<ConstraintSet>(ConstraintSet());
}


void System::set_pins(
        const std::vector<int> &inds,
        const std::vector<math::Vec2> &points) {
    int n_pins = inds.size();
    const int dof = 2 * m_X.rows();
    bool pin_in_place = static_cast<int>(points.size()) != n_pins;
    if ( (dof == 0 && pin_in_place) || (pin_in_place && points.size() > 0) ) {
        throw std::runtime_error("**Solver::set_pins Error: Bad input.");
    }

    m_constraints->m_pins.clear();
    for ( int i = 0; i < n_pins; ++i ) {
        int idx = inds[i];
        if ( pin_in_place ) {
            m_constraints->m_pins[idx] = m_X.row(idx);
        } else {
            m_constraints->m_pins[idx] = points[i];
        }
    }
}


std::shared_ptr<ConstraintSet> System::constraint_set() {
    return m_constraints;
}


bool System::possibly_update_weights(double gamma) {
    const int n_tri_elements = num_tri_elements();
    double do_reweight = false;

    if (n_tri_elements > 0) {
        double max_wsq_ratio = m_tri_elements->max_wsq_ratio();
        do_reweight = (max_wsq_ratio > gamma);
        if (do_reweight) {
            m_tri_elements->update_actual_weights();
        }
    }

    return do_reweight;
}


double System::penalty_energy(bool lagged_s, bool lagged_u) const {
    double penalty = 0.;

    const int n_tri_elements = num_tri_elements();

    if (n_tri_elements > 0) {
        penalty += m_tri_elements->penalty_energy(lagged_s, lagged_u);
    }

    return penalty;
}


void System::midupdate_WtW(const math::MatX2& X) {
    const int n_tri_elements = num_tri_elements();

    if (n_tri_elements > 0) {
        m_tri_elements->midupdate_weights(X);
    }
}


void System::midupdate_WtW() {

    const int n_tri_elements = num_tri_elements();

    if (n_tri_elements > 0) {
        m_tri_elements->midupdate_weights();
    }
}


std::vector<double> System::get_distortion() const {
    const int n_tri_elements = num_tri_elements();
    if (n_tri_elements > 0) {
        return m_tri_elements->get_distortion();
    } else {
        throw std::runtime_error("Error: There are no triangle elements. Cannot return element distortions.");
    }
}

math::MatX2 System::get_b() {
    if (num_tri_elements() == 0) { 
        throw std::runtime_error("Error: There are no triangle elements.");
    }
    return m_tri_elements->get_b();
}

void System::get_A(math::Triplets &triplets) {
    const int n_tri_elements = num_tri_elements();
    if (n_tri_elements > 0) {
        m_tri_elements->get_A_triplets(triplets);
    }
}

void System::update_A(math::SpMat &A) {
    // Note: Do not setZero here! Space has already been reserved, no zeroing is needed.
    const int n_tri_elements = num_tri_elements();
    if (n_tri_elements > 0) {
        m_tri_elements->update_A_coeffs(A);
    }    
}

int System::inversion_count() const {
    int inv_count = 0;
    const int n_tri_elements = num_tri_elements();
    if (n_tri_elements > 0) {
        inv_count += m_tri_elements->inversion_count();
    }
    return inv_count;    
}

void System::unaware_local_update(bool update_candidate_weights) {
    const int n_tri_elements = num_tri_elements();

    if (n_tri_elements > 0) {
        m_tri_elements->update(update_candidate_weights);
    }
}


double System::rotaware_local_update(bool update_candidate_weights) {
    double delta_auglag = 0;

    const int n_tri_elements = num_tri_elements();
    if (n_tri_elements > 0) {
        delta_auglag += m_tri_elements->polar_update(update_candidate_weights);
    }

    return delta_auglag;
}


void System::update_fix_cache(const math::MatX2 &x_fix) {
    const int n_tri_elements = num_tri_elements();

    if (n_tri_elements > 0) {
        m_tri_elements->update_fix_cache(x_fix);
    }
}

void System::update_defo_cache(const math::MatX2 &x_free, bool compute_polar_data) {
    const int n_tri_elements = num_tri_elements();

    if (n_tri_elements > 0) {
        m_tri_elements->update_defo_cache(x_free, compute_polar_data);
    }
}

void System::initialize_dual_vars(const math::MatX2& X) {
    const int n_tri_elements = num_tri_elements();

    if (n_tri_elements > 0) {
        m_tri_elements->initialize_dual_vars(X);
    }
}

void System::advance_dual_vars() {
    const int n_tri_elements = num_tri_elements();

    if (n_tri_elements > 0) {
        m_tri_elements->advance_dual_vars();
    }    
}


double System::potential_energy() const {
    double pe = 0.;

    const int n_tri_elements = num_tri_elements();

    if (n_tri_elements > 0) {
        pe += m_tri_elements->potential_energy_x();
    }

    return pe;
}

double System::potential_gradnorm() const {
    double gradnorm = 1.e30;

    const int n_tri_elements = num_tri_elements();

    if (n_tri_elements > 0) {
        gradnorm = m_tri_elements->potential_gradnorm_x();
    }

    return gradnorm;    
}

double System::dual_objective() const {
    double pe = 0.;
    const int n_tri_elements = num_tri_elements();

    if (n_tri_elements > 0) {
        pe += m_tri_elements->dual_potential_energy();
    }

    return pe;    
}

double System::global_obj_value(
        const math::MatX2 &x_free) {
    double value = 0.;

    const int n_tri_elements = num_tri_elements();

    if (n_tri_elements > 0) {
        value += m_tri_elements->global_obj_value(x_free);
    }

    return value;
}

double System::global_obj_grad(
        const math::MatX2 &x_free,
        math::MatX2 &grad) {

    double value = 0.;
    grad.resize(x_free.rows(), 2);
    grad.setZero();

    const int n_tri_elements = num_tri_elements();

    if (n_tri_elements > 0) {
        math::MatX2 grad_tris;
        value += m_tri_elements->global_obj_grad(x_free, grad_tris);
        grad += grad_tris;
    }

    return value;
}


void System::global_obj_grad(
        math::MatX2 &grad) {
    const int n_tri_elements = num_tri_elements();

    if (n_tri_elements > 0) {
        math::MatX2 grad_tris;
        m_tri_elements->global_obj_grad(grad_tris);
        grad += grad_tris;        
    }
}


}  // namespace wrapd

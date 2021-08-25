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

#include <algorithm>
#include <vector>
#include <iomanip>

#include "TriElements.hpp"
#include "MCL/MicroTimer.hpp"

namespace wrapd {

//
// TriElements
//

TriElements::TriElements(
        const std::vector<math::Vec3i> &tri,
        const std::vector<math::Vec3i> &free_index_tri,
        const std::vector<math::Vec3i> &fix_index_tri,
        const std::vector<std::vector<math::Vec3> > &verts,
        const Settings &settings) :
        m_num_elements(tri.size()),
        m_tri(tri),
        m_free_index_tri(free_index_tri),
        m_fix_index_tri(fix_index_tri),
        m_settings(settings),
        m_reweighting(settings.m_reweighting) {

    m_weight_squared.resize(m_num_elements);
    m_cand_weight_squared.resize(m_num_elements);
    m_elastic_prox_ls_method = mcl::optlib::LSMethod::BacktrackingCubic;

    m_area.resize(m_num_elements);
    m_rest_pose.resize(m_num_elements);
    m_Di_local.resize(m_num_elements);
    m_Dfree_local.resize(m_num_elements);
    m_Dfix_local.resize(m_num_elements);

    m_Dfix_xfix_local.resize(m_num_elements);
    m_Dt_D_local.resize(m_num_elements);
    m_last_sigma.resize(m_num_elements);

    m_Zi.resize(m_num_elements);
    m_prev_Zi.resize(m_num_elements);
    m_Ui.resize(m_num_elements);
    m_prev_Ui.resize(m_num_elements);

    m_cached_F.resize(m_num_elements);
    m_cached_symF.resize(m_num_elements);
    m_cached_svdsigma.resize(m_num_elements);
    m_cached_svdU.resize(m_num_elements);
    m_cached_svdV.resize(m_num_elements);
    m_num_free_inds.resize(m_num_elements);

    double k = 8.;  // Symmetric Dirichlet stiffness parameter

    double beta_static = m_settings.beta_static();

    #pragma omp parallel for
    for (int e = 0; e < m_num_elements; e++) {
        math::Mat3x2 edges;
        edges.col(0) = verts[e][1] - verts[e][0];
        edges.col(1) = verts[e][2] - verts[e][0];

        math::Vec3 n1 = edges.col(0).normalized();
        math::Mat3x2 basis;
        basis.col(0) = n1;
        basis.col(1) = (edges.col(1)-edges.col(1).dot(n1)*n1).normalized();

        m_rest_pose[e] = (basis.transpose() * edges).inverse();
        m_area[e] = (basis.transpose() * edges).determinant() / 2.0;

        math::Mat3x2 S;
        S.setZero();
        S(0, 0) = -1;
        S(0, 1) = -1;
        S(1, 0) =  1;
        S(2, 1) =  1;
        math::Mat3x2 D = S * m_rest_pose[e];
        m_Di_local[e] = D.transpose();
        m_Dt_D_local[e] = D * D.transpose();
        m_num_free_inds[e] = 0;
        for (int i = 0; i < 3; i++) {
            if (m_free_index_tri[e][i] >= 0) {
                m_num_free_inds[e] += 1;
            }
        }
        m_Dfree_local[e].resize(2, m_num_free_inds[e]);
        m_Dfix_xfix_local[e].setZero();
        m_Dfix_local[e].resize(2, 3 - m_num_free_inds[e]);
        int j = 0; 
        for (int i = 0; i < 3; i++) {
            if (m_free_index_tri[e][i] >= 0) {
                m_Dfree_local[e].col(j) = m_Di_local[e].col(i);
                j += 1;
            }
        }        

        j = 0;
        for (int i = 0; i < 3; i++) {
            if (m_fix_index_tri[e][i] >= 0) {
                m_Dfix_local[e].col(j) = m_Di_local[e].col(i);
                j += 1;
            }
        }


        m_weight_squared[e] = beta_static * k * m_area[e];
        m_last_sigma[e] = math::Vec2::Ones();
    }

}


math::MatX2 TriElements::get_b() const {
    math::MatX2 b = math::MatX2::Zero(m_xfree_rows, 2);
    for (int e = 0; e < m_num_elements; e++) {
        math::Mat3x2 b_local = m_weight_squared[e] * m_Di_local[e].transpose() * (m_Zi[e] - m_Ui[e] - m_Dfix_xfix_local[e]);
        for (int i = 0; i < 3; i++) {
            if (m_free_index_tri[e][i] >= 0) {
                b.row(m_free_index_tri[e][i]) += b_local.row(i);
            }
        }
    }
    return b;
}

void TriElements::get_A_triplets(math::Triplets &triplets) {
    triplets.resize(9*m_num_elements);
    #pragma omp parallel for
    for (int e = 0; e < m_num_elements; e++) {
        for (int i = 0; i < 3; i++) {
            if (m_free_index_tri[e][i] >= 0) {
                for (int j = 0; j < 3; j++) {
                    if (m_free_index_tri[e][j] >= 0 && m_free_index_tri[e][i] <= m_free_index_tri[e][j]) {
                        triplets[9*e + 3*i + j] = math::Triplet(m_free_index_tri[e][i], m_free_index_tri[e][j], m_weight_squared[e] * m_Dt_D_local[e](i, j));
                    } else {
                        triplets[9*e + 3*i + j] = math::Triplet(0, 0, 0);
                    }
                }
            } else {
                triplets[9*e + 3*i] = math::Triplet(0, 0, 0.);
            }
        }
    }
}

void TriElements::update_A_coeffs(math::SpMat &A) {
    for (int e = 0; e < m_num_elements; e++) {
        for (int i = 0; i < 3; i++) {
            if (m_free_index_tri[e][i] >= 0) {
                for (int j = 0; j < 3; j++) {
                    if (m_free_index_tri[e][j] >= 0 && m_free_index_tri[e][i] <= m_free_index_tri[e][j]) {
                        A.coeffRef(m_free_index_tri[e][i], m_free_index_tri[e][j]) += m_weight_squared[e] * m_Dt_D_local[e](i, j);
                    }
                }
            }
        }
    }
}

void TriElements::initialize_dual_vars(const math::MatX2& X) {

    if (m_settings.m_rot_awareness == Settings::RotAwareness::ENABLED) {
        #pragma omp parallel for
        for (int e = 0; e < m_num_elements; e++) {
            math::Mat3x2 x_local;
            x_local.row(0) = X.row(m_tri[e][0]);
            x_local.row(1) = X.row(m_tri[e][1]);
            x_local.row(2) = X.row(m_tri[e][2]);
            math::Mat2x2 Ft = m_Di_local[e] * x_local;
            math::Mat2x2 F = Ft.transpose();
            math::Mat2x2 R;
            math::Mat2x2 S;
            math::polar(F, R, S);

            m_Zi[e] = S.transpose();
            m_Ui[e].setZero();
            m_prev_Zi[e] = m_Zi[e];
            m_prev_Ui[e] = m_Ui[e];
        }
        m_curr_pe = dual_potential_energy();
        m_prev_pe = m_curr_pe;
    } else if (m_settings.m_rot_awareness == Settings::RotAwareness::DISABLED) {
        #pragma omp parallel for
        for (int e = 0; e < m_num_elements; e++) {
            math::Mat3x2 x_local;
            x_local.row(0) = X.row(m_tri[e][0]);
            x_local.row(1) = X.row(m_tri[e][1]);
            x_local.row(2) = X.row(m_tri[e][2]);
            m_Zi[e] = m_Di_local[e] * x_local;
            m_Ui[e].setZero();
            m_prev_Zi[e] = m_Zi[e];
            m_prev_Ui[e] = m_Ui[e];
        }
    }
}

void TriElements::advance_dual_vars() {
    #pragma omp parallel for
    for (int e = 0; e < m_num_elements; e++) {
        m_prev_Zi[e] = m_Zi[e];
        m_prev_Ui[e] = m_Ui[e];
    }
}

void TriElements::update(bool update_candidate_weights) {

    std::vector<math::Vec2> temp_sigmas(m_num_elements);

    #pragma omp parallel
    {
        mcl::optlib::Newton<double, 2> solver;
        solver.m_settings.ls_method = m_elastic_prox_ls_method;
        solver.m_settings.max_iters = 2000;
        SDProx problem;
            
        #pragma omp for
        for (int e = 0; e < m_num_elements; e++) {
            math::Mat2x2 Di_x = m_cached_F[e].transpose();
            math::Mat2x2 Ft = Di_x + m_Ui[e];

            math::Mat2x2 F = Ft.transpose();
            math::Vec2 sigma;
            math::Mat2x2 U;
            math::Mat2x2 V;
            math::svd(F, sigma, U, V, true);
                
            problem.set_x0(sigma);
            problem.set_wsq_over_area(m_weight_squared[e] / m_area[e]);
            sigma = m_last_sigma[e];
            solver.minimize(problem, sigma);
            m_last_sigma[e] = sigma;

            if (sigma[0] < 0. || sigma[1] < 0.) {
                printf("Negative singular value in solution. svals: %f %f\n", sigma[0], sigma[1]);                    
                exit(0);
            }

            Ft.noalias() = V * sigma.asDiagonal() * U.transpose();

            m_Ui[e] += (Di_x - Ft);
            m_Zi[e] = Ft;
            temp_sigmas[e] = sigma;
        }
    }

    if (update_candidate_weights) {
        update_all_candidate_weights(temp_sigmas);
    }

}

double TriElements::polar_update(bool update_candidate_weights) {
    (void)(update_candidate_weights);

    m_prev_pe = m_curr_pe;
    m_curr_pe = 0.;
    double bef_spec = 0.;
    double aft_spec = 0.;

    std::vector<math::Vec2> temp_sigmas(m_num_elements);

    static int curr_admm_iter = 0;

    #pragma omp parallel
    {
        mcl::optlib::Newton<double, 2> solver;
        solver.m_settings.ls_method = m_elastic_prox_ls_method;
        solver.m_settings.max_iters = 2000;
        SDProx problem;

        #pragma omp for reduction ( + : m_curr_pe, bef_spec, aft_spec)
        for (int e = 0; e < m_num_elements; e++) {
            math::Mat2x2 S_inout = m_cached_symF[e] + m_Ui[e]; // don't need to use S.transpose() since guaranteed symmetric

            bef_spec += 0.5 * m_weight_squared[e] * (m_cached_symF[e] - (m_Zi[e] - m_Ui[e])).squaredNorm();

            math::Mat2x2 S = S_inout.transpose();
            math::Vec2 sigma;
            math::Mat2x2 U;
            math::Mat2x2 V;
            math::svd(S, sigma, U, V, true);
            problem.set_x0(sigma);
            problem.set_wsq_over_area(m_weight_squared[e] / m_area[e]);
            sigma = m_last_sigma[e];
            solver.minimize(problem, sigma);
            temp_sigmas[e] = sigma;
            m_last_sigma[e] = sigma;

            if (sigma[0] < 0. || sigma[1] < 0.) {
                printf("Negative singular value in solution. svals: %f %f:\n", sigma[0], sigma[1]);                    
                exit(0);
            }

            S_inout.noalias() = V * sigma.asDiagonal() * U.transpose();

            double val = 0.;
            for (int i = 0; i < 2; ++i) {
                val += (sigma[i] * sigma[i] + std::pow(sigma[i], -2.0));
            }
            m_curr_pe += m_area[e] * (val - 4.0);

            aft_spec += 0.5 * m_weight_squared[e] * (m_cached_symF[e] - (S_inout - m_Ui[e])).squaredNorm();

            m_Zi[e] = S_inout;
            m_Ui[e] += m_cached_symF[e] - S_inout;    
        }
    }
    curr_admm_iter++;

    if (update_candidate_weights) {
        update_all_candidate_weights(temp_sigmas);
    }

    return (aft_spec - bef_spec + m_curr_pe - m_prev_pe);
}

void TriElements::midupdate_weights(const math::MatX2 &X) {
    if (m_reweighting == Settings::Reweighting::ENABLED) {
        std::vector<math::Vec2> temp_sigmas(m_num_elements);
        #pragma omp parallel for
        for (int e = 0; e < m_num_elements; e++) {
            math::Mat3x2 x_local;
            x_local.row(0) = X.row(m_tri[e][0]);
            x_local.row(1) = X.row(m_tri[e][1]);
            x_local.row(2) = X.row(m_tri[e][2]);
            math::Mat2x2 Ft = m_Di_local[e] * x_local;
            math::Mat2x2 F = Ft.transpose();
            math::Mat2x2 U;
            math::Mat2x2 V;
            math::svd(F, temp_sigmas[e], U, V, true);
        }
        update_all_candidate_weights(temp_sigmas);
        update_actual_weights();
    }
}

void TriElements::midupdate_weights() {
    if (m_reweighting == Settings::Reweighting::ENABLED) {
        update_all_candidate_weights(m_cached_svdsigma);
        update_actual_weights();
    }
}


void TriElements::update_all_candidate_weights(const std::vector<math::Vec2>& sigmas) {
    if (m_reweighting == Settings::Reweighting::ENABLED) {
        double wsq_rest = 8.0;

        static int count = 0;
        if (count <= 1) { 
            #pragma omp parallel for
            for (int e = 0; e < m_num_elements; e++) {
                m_cand_weight_squared[e] = m_area[e] * wsq_rest;    
            }
        } else { 

            if (count <= 200 && m_settings.weight_clamp_easing()) {   
                static double extramult = 1.0;     
                const double wsq_min = wsq_rest * m_settings.beta_min();
                const double wsq_max = wsq_rest * std::min(10.0 * extramult, m_settings.beta_max());
                extramult *= 1.5000001;  // slightly over 1.5 so that gamma=1.5 works as expected

                #pragma omp parallel for
                for (int e = 0; e < m_num_elements; e++) {
                    double H00 = (2.0 + 6.0 * std::pow(sigmas[e][0], -4.0));
                    double H11 = (2.0 + 6.0 * std::pow(sigmas[e][1], -4.0));
                    double cand_stiffness = math::clamp(wsq_min, std::max(H00, H11), wsq_max);

                    m_cand_weight_squared[e] = m_area[e] * cand_stiffness;     
                }
            } else {
             
                const double wsq_min = wsq_rest * m_settings.beta_min();
                const double wsq_max = wsq_rest * m_settings.beta_max();

                #pragma omp parallel for
                for (int e = 0; e < m_num_elements; e++) {
                    double H00 = (2.0 + 6.0 * std::pow(sigmas[e][0], -4.0));
                    double H11 = (2.0 + 6.0 * std::pow(sigmas[e][1], -4.0));
                    double cand_stiffness = math::clamp(wsq_min, std::max(H00, H11), wsq_max);

                    m_cand_weight_squared[e] = m_area[e] * cand_stiffness;     
                }                
            }
        }
        count++;
    }
}

double TriElements::max_wsq_ratio() {
    double max_ratio = 0.;
    for (int e = 0; e < m_num_elements; e++) {
        double currsq_over_candsq = m_weight_squared[e] / m_cand_weight_squared[e];
        double candsq_over_currsq = 1.0 / currsq_over_candsq;
        if (currsq_over_candsq > max_ratio) {
            max_ratio = currsq_over_candsq;
        }
        if (candsq_over_currsq > max_ratio) {
            max_ratio = candsq_over_currsq;
        }
    }
    return max_ratio;
}

void TriElements::update_actual_weights() {
    #pragma omp parallel for
    for (int e = 0; e < m_num_elements; e++) {
        m_prev_Ui[e] = (m_weight_squared[e] / m_cand_weight_squared[e]) * m_prev_Ui[e];
        m_Ui[e] = (m_weight_squared[e] / m_cand_weight_squared[e]) * m_Ui[e];
        m_weight_squared[e] = m_cand_weight_squared[e];
    }
}

int TriElements::inversion_count() const {
    int count = 0;

    if (m_settings.m_rot_awareness == Settings::RotAwareness::ENABLED) {
        #pragma omp parallel for reduction (+:count)
        for (int e = 0; e < m_num_elements; e++) {
            if (m_cached_svdsigma[e][0] < 0. || m_cached_svdsigma[e][1] < 0.) {
                count += 1;
            }
        }
    } else if (m_settings.m_rot_awareness == Settings::RotAwareness::DISABLED) {
        #pragma omp parallel for reduction (+:count)
        for (int e = 0; e < m_num_elements; e++) {
            math::Vec2 sigma;
            math::Mat2x2 U;
            math::Mat2x2 V;
            math::svd(m_cached_F[e], sigma, U, V, true);             
            if (sigma[0] < 0. || sigma[1] < 0.) {
                count += 1;
            }
        }
    }

    return count;
}

double TriElements::penalty_energy(bool lagged_s, bool lagged_u) const {
    double penalty = 0.;
    #pragma omp parallel for reduction( + : penalty )
    for (int e = 0; e < m_num_elements; e++) {
        math::Mat2x2 SymF = m_cached_symF[e];
        if (lagged_u && lagged_s) {
            penalty += 0.5 * m_weight_squared[e] * (SymF - (m_prev_Zi[e] - m_prev_Ui[e])).squaredNorm();
        } else if (lagged_u) {
            penalty += 0.5 * m_weight_squared[e] * (SymF - (m_Zi[e] - m_prev_Ui[e])).squaredNorm();
        } else {
            penalty += 0.5 * m_weight_squared[e] * (SymF - (m_Zi[e] - m_Ui[e])).squaredNorm();
        }
    }
    return penalty; 
}


void TriElements::update_fix_cache(const math::MatX2 &x_fix) {
    #pragma omp parallel for
    for (int e = 0; e < m_num_elements; e++) {
        if (m_num_free_inds[e] < 3) {
            math::MatX2 x_fix_local(3 - m_num_free_inds[e], 2);
            int j = 0;
            for (int i = 0; i < 3; i++) {
                if (m_fix_index_tri[e][i] >= 0) {
                    x_fix_local.row(j) = x_fix.row(m_fix_index_tri[e][i]);
                    j += 1;
                }
            }
            m_Dfix_xfix_local[e] = m_Dfix_local[e] * x_fix_local;
        }
    }
}

void TriElements::update_defo_cache(const math::MatX2 &x_free, bool compute_polar_data) {
    m_xfree_rows = x_free.rows();
    const bool should_compute_polar_data = compute_polar_data;
    #pragma omp parallel for
    for (int e = 0; e < m_num_elements; e++) {
        if (m_num_free_inds[e] == 3) {
            math::Mat3x2 x_free_local;
            x_free_local.row(0) = x_free.row(m_free_index_tri[e][0]);
            x_free_local.row(1) = x_free.row(m_free_index_tri[e][1]);
            x_free_local.row(2) = x_free.row(m_free_index_tri[e][2]);
            m_cached_F[e] = (m_Di_local[e] * x_free_local).transpose();
        } else {
            math::MatX2 x_free_local(m_num_free_inds[e], 2);
            int j = 0;
            for (int i = 0; i < 3; i++) {
                if (m_free_index_tri[e][i] >= 0) {
                    x_free_local.row(j) = x_free.row(m_free_index_tri[e][i]);
                    j += 1;
                }
            }
            m_cached_F[e] = ((m_Dfree_local[e] * x_free_local) + m_Dfix_xfix_local[e]).transpose();
        }
        if (should_compute_polar_data) {
            math::svd(m_cached_F[e], m_cached_svdsigma[e], m_cached_svdU[e], m_cached_svdV[e]);
            m_cached_symF[e] = m_cached_svdV[e] * m_cached_svdsigma[e].asDiagonal() * m_cached_svdV[e].transpose();
        }
    }
    
}

double TriElements::potential_energy_x() const {
    double pe = 0.;

    #pragma omp parallel for reduction ( + : pe)
    for (int e = 0; e < m_num_elements; e++) {
        math::Vec2 sigma = m_cached_svdsigma[e];          
        if (sigma[0] < 0. || sigma[1] < 0.) {
            pe += 1.e30;
        } else {
            double val = 0.;
            for (int i = 0; i < 2; ++i) {
                val += (sigma[i] * sigma[i] + std::pow(sigma[i], -2.0));
            }
            pe += m_area[e] * (val - 4.0);
        }
    }        

    return pe;
}


double TriElements::potential_gradnorm_x() const {

    math::MatX2 potential_grad = math::MatX2::Zero(m_xfree_rows, 2);
    #pragma omp parallel
    {
        math::MatX2 gradient_th = math::MatX2::Zero(m_xfree_rows, 2);
        #pragma omp for
        for (int e = 0; e < m_num_elements; e++) {
            math::Vec2 sigma = m_cached_svdsigma[e];
            math::Mat2x2 U = m_cached_svdU[e];
            math::Mat2x2 V = m_cached_svdV[e];        

            math::Vec2 pow_sig;
            pow_sig[0] = std::pow(sigma[0], -3.0);
            pow_sig[1] = std::pow(sigma[1], -3.0);
            math::Vec2 grad2 = 2.0 * (sigma - pow_sig); 

            math::Mat2x2 P;
            if (sigma[0] < 0 || sigma[1] < 0) {
                P = U * math::Vec2(1.e30, 1.e30).asDiagonal() * V.transpose();
            } else {
                P = U * grad2.asDiagonal() * V.transpose();
            }

            math::Mat3x2 g = m_Di_local[e].transpose() * (m_area[e] * P.transpose());
            for (int ii = 0; ii < 3; ii++) {
                if (m_free_index_tri[e][ii] != -1) {
                    gradient_th.row(m_free_index_tri[e][ii]) += g.row(ii);
                }
            }                
        }

        #pragma omp critical
        {
            potential_grad += gradient_th;
        }
    }
    return potential_grad.norm();
}


double TriElements::dual_potential_energy() const {
    double pe = 0.;

    #pragma omp parallel for reduction ( + : pe)
    for (int e = 0; e < m_num_elements; e++) {
        math::Mat2x2 Zi_t = m_Zi[e].transpose();  // F or S depending on method
        math::Vec2 sigma;
        math::Mat2x2 U;
        math::Mat2x2 V;
        math::svd(Zi_t, sigma, U, V, true);             
        if (sigma[0] < 0. || sigma[1] < 0.) {
            pe += 1.e30;
        } else {
            double val = (sigma[0] * sigma[0] + std::pow(sigma[0], -2.0))
                    + (sigma[1] * sigma[1] + std::pow(sigma[1], -2.0));
            pe += m_area[e] * (val - 4.0);
        }
    }

    return pe;    
}


std::vector<double> TriElements::get_distortion() const {
    std::vector<double> distortion(m_num_elements);
    #pragma omp parallel for
    for (int e = 0; e < m_num_elements; e++) {     
        if (m_cached_svdsigma[e][0] < 0. || m_cached_svdsigma[e][1] < 0.) {
            distortion[e] = 1.e8;
        } else {
            double val = 0.25 * (std::pow(m_cached_svdsigma[e][0], 2.0) + std::pow(m_cached_svdsigma[e][0], -2.0)
                    + std::pow(m_cached_svdsigma[e][1], 2.0) + std::pow(m_cached_svdsigma[e][1], -2.0));
            if (val < 1.0) { 
                distortion[e] = 0.;
            } else {
                distortion[e] = std::log(val);
            }
        }
    }
    return distortion;
}



double TriElements::global_obj_value(const math::MatX2 &x_free) {
    double val = 0.;

    update_defo_cache(x_free);

    #pragma omp parallel for reduction ( + : val )
    for (int e = 0; e < m_num_elements; e++) {
        math::Mat2x2 P = (m_Zi[e] - m_Ui[e]).transpose();
        math::Mat2x2 SymF = m_cached_svdV[e] * m_cached_svdsigma[e].asDiagonal() * m_cached_svdV[e].transpose();
        val += 0.5 * m_weight_squared[e] * (SymF - P).squaredNorm();
    }
    
    return val;    
}


double TriElements::global_obj_grad(const math::MatX2 &x_free, math::MatX2 &gradient) {
    
    update_defo_cache(x_free);
    global_obj_grad(gradient);

    double val = 0.;
    #pragma omp parallel for reduction ( + : val )
    for (int e = 0; e < m_num_elements; e++) {
        math::Mat2x2 P = (m_Zi[e] - m_Ui[e]).transpose();
        val += 0.5 * m_weight_squared[e] * (m_cached_symF[e] - P).squaredNorm();
    }
    
    return val;  
}


void TriElements::global_obj_grad(math::MatX2 &gradient) {
    std::vector<math::Mat2x3> gradients_transpose(m_num_elements);

    #pragma omp parallel for
    for (int e = 0; e < m_num_elements; e++) {
        math::Vec2 s = m_cached_svdsigma[e];
        math::Mat2x2 Q = m_cached_svdV[e].transpose() * (m_Zi[e] - m_Ui[e]).transpose() * m_cached_svdV[e];

        math::Mat2x2 K_plus_Kt;
        K_plus_Kt(0, 0) = Q(0, 0) / s[0];
        K_plus_Kt(0, 1) = K_plus_Kt(1, 0) = (Q(0, 1) + Q(1, 0)) / (s[0] + s[1]);
        K_plus_Kt(1, 1) = Q(1, 1) / s[1];

        gradients_transpose[e].noalias() = m_weight_squared[e] * (m_cached_F[e] - (m_cached_svdU[e] * s.asDiagonal()
                * K_plus_Kt * m_cached_svdV[e].transpose())) * m_Di_local[e];
    }

    gradient.resize(m_xfree_rows, 2);
    gradient.setZero();
    for (int e = 0; e < m_num_elements; e++) {
        for (int ii = 0; ii < 3; ii++) {
            if (m_free_index_tri[e][ii] == -1) {
                continue;
            }
            gradient.row(m_free_index_tri[e][ii]) += gradients_transpose[e].col(ii);
        }
    }

}


}  // namespace wrapd

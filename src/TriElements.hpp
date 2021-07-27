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

#ifndef SRC_TRIELEMENTS_HPP_
#define SRC_TRIELEMENTS_HPP_

#include <memory>
#include <vector>

#include "MCL/Newton.hpp"

#include "Math.hpp"
#include "Settings.hpp"

namespace wrapd {

class SDProx : public mcl::optlib::Problem<double, 2> {
 public:
    void set_x0(const math::Vec2 x0) { m_x0 = x0; }

    virtual bool converged(const math::Vec2 &x0, const math::Vec2 &x1, const math::Vec2 &grad) {
        return (grad.norm() < 1e-12 || (x0-x1).norm() < 1e-8);
    }

    void set_wsq_over_area(double wsq_over_area) { m_wsq_over_area = wsq_over_area; }

    double energy_density(const math::Vec2 &x) const {
        return (x[0] * x[0]) + std::pow(x[0], -2.0) + (x[1] * x[1]) + std::pow(x[1], -2.0);
    }

    double physical_value(const math::Vec2 &x) {
        if (x[0] < 0. || x[1] < 0.) {
            return 1.e30;
        }
        return energy_density(x) - 4.0;
    }

    double value(const math::Vec2 &x) {
        if (x[0] < 0. || x[1] < 0.) {
            return 1.e30;
        }
        double t1 = energy_density(x);
        double t2 = 0.5 * m_wsq_over_area * (x - m_x0).squaredNorm();  // quad penalty
        return t1 + t2;
    }

    double gradient(const math::Vec2 &x, math::Vec2 &grad) {
        math::Vec2 pow_x;
        pow_x[0] = std::pow(x[0], -3.0);
        pow_x[1] = std::pow(x[1], -3.0);
        grad = 2.0 * (x - pow_x) + m_wsq_over_area * (x - m_x0);
        return value(x);
    }

    void hessian(const math::Vec2 &x, math::Mat2x2 &hess) {
        hess(0, 1) = hess(1, 0) = 0.;
        hess(0, 0) = (2.0 + 6.0 * std::pow(x[0], -4.0)) + m_wsq_over_area;
        hess(1, 1) = (2.0 + 6.0 * std::pow(x[1], -4.0)) + m_wsq_over_area;
    }

    void solve_hessian(const math::Vec2 &x, const math::Vec2 &grad, math::Vec2 &dx) {
        dx[0] = -grad[0] / ((2.0 + 6.0 * std::pow(x[0], -4.0)) + m_wsq_over_area);
        dx[1] = -grad[1] / ((2.0 + 6.0 * std::pow(x[1], -4.0)) + m_wsq_over_area);
    }

 private:
    math::Vec2 m_x0;
    double m_wsq_over_area;
};

class TriElements {
 public:

    TriElements(
            const std::vector<math::Vec3i> &tri,
            const std::vector<math::Vec3i> &free_index_tri,
            const std::vector<math::Vec3i> &fix_index_tri,
            const std::vector< std::vector<math::Vec3> > &verts,
            const Settings &settings);

    int num_elements() const { return m_num_elements; }

    math::MatX2 get_b() const;
    void get_A_triplets(math::Triplets &triplets);
    void update_A_coeffs(math::SpMat& A);

    double penalty_energy(bool lagged_s, bool lagged_u) const;

    void update_fix_cache(const math::MatX2 &x_fix);
    void update_defo_cache(const math::MatX2 &x_free, bool compute_polar_data = true);

    double potential_energy_x() const;  // as a function of x
    double potential_gradnorm_x() const;  // as a function of x

    double dual_potential_energy() const;  // as a function of Z (so F or sym(F), depending on method)

    void initialize_dual_vars(const math::MatX2& X);
    void advance_dual_vars();

    void update(bool update_candidate_weights);
    double polar_update(bool update_candidate_weights);

    void update_all_candidate_weights(const std::vector<math::Vec2>& temp_sigmas);

    double max_weight_discrepancy();
    double max_wsq_ratio();

    void update_actual_weights();

    double global_obj_value(const math::MatX2 &x_free);
    double global_obj_grad(const math::MatX2 &x_free, math::MatX2 &gradient);
    void global_obj_grad(math::MatX2 &gradient);

    void midupdate_weights(const math::MatX2 &X);
    void midupdate_weights();

    int inversion_count() const;

    const std::vector<math::Vec3i>& inds() const { return m_tri; }

 protected:

    const int m_num_elements;
    std::vector<double> m_weight_squared;
    std::vector<double> m_cand_weight_squared;

    double m_prev_pe;  // past local step dual potential energy
    double m_curr_pe; // most recent local step dual potential energy


    /* 3-tuples of vertex indices (consistent with ordering of actual mesh vertices) */
    std::vector<math::Vec3i> m_tri; 

    /* 3-tuples of vertex indices. Reindexed -- maps to free vertex list. Free indices are non-zero, fixed are -1. */
    std::vector<math::Vec3i> m_free_index_tri;

    /* 3-tuples of vertex indices. Reindexed -- maps to fixed vertex list. Fixed indices are non-zero, free are -1. */
    std::vector<math::Vec3i> m_fix_index_tri;

    const Settings& m_settings;

    /* The areas of all zero-distortion reference triangles */
    std::vector<double> m_area;

    /* The reference bases for each triangle */
    std::vector<math::Mat2x2> m_rest_pose;

    const Settings::Reweighting m_reweighting;

    /* Reduction matrices that map position coordinates to the deformation gradient */
    std::vector<math::Mat2x3> m_Di_local;
    std::vector<math::Mat2X> m_Dfree_local;
    std::vector<math::Mat2X> m_Dfix_local;
    
    std::vector<math::Mat2x2> m_Dfix_xfix_local;
    std::vector<math::Mat3x3> m_Dt_D_local;
    
    /* Singular values solution from the previous ADMM iteration */
    std::vector<math::Vec2> m_last_sigma;

    /* Dual variables used during optimization. */
    std::vector<math::Mat2x2> m_Zi;
    std::vector<math::Mat2x2> m_prev_Zi;
    std::vector<math::Mat2x2> m_Ui;
    std::vector<math::Mat2x2> m_prev_Ui;

    /* Cached data for rotation-aware solver - be careful to ensure the cache isn't stale! */
    std::vector<math::Mat2x2> m_cached_F;
    std::vector<math::Mat2x2> m_cached_symF;
    std::vector<math::Vec2> m_cached_svdsigma;
    std::vector<math::Mat2x2> m_cached_svdU;
    std::vector<math::Mat2x2> m_cached_svdV;

    /* Number of free rows in the global position vector X */
    int m_xfree_rows;

    std::vector<int> m_num_free_inds;
    mcl::optlib::LSMethod m_elastic_prox_ls_method;

    static inline double clamp(double input_val, double min_val, double max_val) {
        return std::min(std::max(input_val, min_val), max_val);
    }

};  // end class TriElements


template <typename IN_SCALAR, typename TYPE>
inline void create_tris_from_mesh(
        std::shared_ptr<TriElements>& tri_elements,
        const IN_SCALAR *verts,
        const int *inds,
        int n_tris,
        const Settings& settings,
        const std::vector<int> &free_index_list,
        const std::vector<int> &fix_index_list,
        const int vertex_offset) {

    std::vector<math::Vec3i> tris(n_tris);
    std::vector<math::Vec3i> free_index_tris(n_tris);
    std::vector<math::Vec3i> fix_index_tris(n_tris);
    std::vector< std::vector<math::Vec3> > tris_verts(n_tris);

    for (int e = 0; e < n_tris; ++e) {
        tris[e] = math::Vec3i(inds[e*3+0], inds[e*3+1], inds[e*3+2]);
        free_index_tris[e] = math::Vec3i(
                free_index_list[inds[e*3+0]],
                free_index_list[inds[e*3+1]],
                free_index_list[inds[e*3+2]]);
        fix_index_tris[e] = math::Vec3i(
                fix_index_list[inds[e*3+0]],
                fix_index_list[inds[e*3+1]],
                fix_index_list[inds[e*3+2]]);
        std::vector<math::Vec3> tri_verts = {
            math::Vec3(verts[tris[e][0]*3+0], verts[tris[e][0]*3+1], verts[tris[e][0]*3+2]),
            math::Vec3(verts[tris[e][1]*3+0], verts[tris[e][1]*3+1], verts[tris[e][1]*3+2]),
            math::Vec3(verts[tris[e][2]*3+0], verts[tris[e][2]*3+1], verts[tris[e][2]*3+2])};
        tris_verts[e] = tri_verts;
        tris[e] += (math::Vec3i(1, 1, 1) * vertex_offset);
    }
    tri_elements = std::make_shared<TYPE>(tris, free_index_tris, fix_index_tris, tris_verts, settings);

}  // end create from mesh


}  // namespace wrapd
#endif  // SRC_TRIELEMENTS_HPP_

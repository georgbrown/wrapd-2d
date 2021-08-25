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

#ifndef SRC_SYSTEM_HPP_
#define SRC_SYSTEM_HPP_

#include <vector>
#include <memory>
#include <unordered_map>

#include "TriElements.hpp"
#include "Settings.hpp"

namespace wrapd {

class ConstraintSet {
 public:
    std::unordered_map<int, math::Vec2> m_pins;  // index -> location

}; 

class System {
 public:
    inline int num_all_verts() const { return m_X.rows(); }

    // Access and modifiy positions

    inline const math::MatX2 X() const {
        return m_X;
    }

    inline const math::MatX2& X_init() const {
        return m_X_init;
    }

    inline void X(const math::MatX2& val) {
        m_X = val;
    }

    template<int DIM>
    inline void X(int idx, math::Mat<DIM, 2> seg) {
        m_X.segment<DIM, 2>(idx, 0) = seg;
    }

    template<int DIM>
    inline const math::Mat<DIM, 2> X(int idx) {
        return m_X.segment<DIM, 2>(idx, 0);
    }

    // Adds nodes to the Solver.
    // Returns the current total number of nodes after insert.
    inline int add_nodes(
            const math::MatX2 &X,
            const math::MatX2 &X_init) {
        const int n_verts = X.rows();
        const int prev_n = m_X.rows();
        const int size = prev_n + n_verts;
        m_X.conservativeResize(size, 2);
        m_X_init.conservativeResize(size, 2);
        for (int i = 0; i < n_verts; ++i) {
            int idx = prev_n + i;
            m_X.row(idx) = X.row(i);
            m_X_init.row(idx) = X_init.row(i);
        }
        return (prev_n + n_verts);
    }

    System(const Settings &settings);

    void set_pins(
            const std::vector<int> &inds,
            const std::vector<math::Vec2> &points);

    std::shared_ptr<ConstraintSet> constraint_set();

    int num_tri_elements() const {
        if (m_tri_elements.get() == nullptr) {
            return 0;
        } else {
            return m_tri_elements->num_elements();
        }
    }

    bool possibly_update_weights(double gamma);

    void midupdate_WtW(const math::MatX2 &curr_x);
    void midupdate_WtW();

    int get_WtW(math::Triplets &triplets);

    std::vector<double> get_distortion() const;

    math::MatX2 get_b();
    void get_A(math::Triplets &triplets);
    void update_A(math::SpMat &A);

    double penalty_energy(bool lagged_s, bool lagged_u) const;

    int inversion_count() const;

    void unaware_local_update(bool update_candidate_weights);
    double rotaware_local_update(bool update_candidate_weights);

    std::shared_ptr<TriElements>& tri_elements() {
        return m_tri_elements;
    }

    void update_fix_cache(const math::MatX2& x_fix);
    void update_defo_cache(const math::MatX2& x_free, bool compute_polar_data);

    void initialize_dual_vars(const math::MatX2& X);
    void advance_dual_vars();

    double potential_energy() const;  // as a function of x
    double potential_gradnorm() const;  // as a function of x

    double dual_objective() const; // as a function of either Z or S

    void set_initial_deformation() {
        m_X = m_X_init;
    }

    double global_obj_value(const math::MatX2 &x_free);
    double global_obj_grad(const math::MatX2 &x_free, math::MatX2 &grad);
    void global_obj_grad(math::MatX2 &grad);

 protected:
    const Settings m_settings;
    std::shared_ptr<TriElements> m_tri_elements;
    std::shared_ptr<ConstraintSet> m_constraints;

 private:
    math::MatX2 m_X;  // node positions
    math::MatX2 m_X_init;  // node positions, initial
    std::vector<math::Vec3i> m_tri_free_inds;
};

}  // namespace wrapd

#endif  // SRC_SYSTEM_HPP_

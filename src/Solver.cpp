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
// Originally by Matt Overby (https://mattoverby.net/)
// Modified by George E. Brown (https://www-users.cse.umn.edu/~brow2327/)


#include <fstream>
#include <unordered_set>
#include <unordered_map>

#include "Solver.hpp"

namespace wrapd {

Solver::Solver(const Settings &settings) : initialized(false) {
    m_settings = settings;
    m_system = std::make_shared<System>(settings);
}

bool Solver::initialize(const Settings &settings) {
    m_settings = settings;
    m_admm_solver.initialize(m_system, m_settings);
    m_system->set_initial_deformation();
    initialized = true;
    return true;
}  // end init


void Solver::set_pins(
        const std::vector<int> &inds,
        const std::vector<math::Vec2> &points ) {
    m_system->set_pins(inds, points);
}

void Solver::solve_offline() {
    math::MatX2 X = m_system->X();
    m_admm_solver.solve(X);
    m_system->X(X);
}

void Solver::setup() {
    math::MatX2 X = m_system->X();
    m_admm_solver.setup_solve(X);
}

void Solver::iterate() {
    m_admm_solver.iterate();
}

bool Solver::termination_check() {
    if (m_admm_solver.algorithm_data()->earlyexit_asap()) {
        return true;
    }
    if (m_admm_solver.algorithm_data()->converged_strong()) {
        return true;
    }
    return false;
}

void Solver::finish() {
    math::MatX2 X;
    m_admm_solver.finish_solve(X);
    m_system->X(X);
}

}  // namespace wrapd

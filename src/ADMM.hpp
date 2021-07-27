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

#ifndef SRC_ADMM_HPP_
#define SRC_ADMM_HPP_

#include <memory>
#include <vector>

#include "Math.hpp"
#include "Settings.hpp"
#include "System.hpp"
#include "AlgorithmData.hpp"
#include "MCL/MicroTimer.hpp"

#include "GlobalStepBehavior.hpp"

namespace wrapd {

class ADMM {
 public:
    ADMM();

    bool initialize(std::shared_ptr<System> system, const Settings& settings);
    void reinitialize() { }

    void prepare();
    void solve(math::MatX2& X);

    AlgorithmData* algorithm_data() { return &m_algorithm_data; }

 private:
    void setup_solve(math::MatX2& x);
    void iterate();
    void finish_solve(math::MatX2& x);

    Settings m_settings;
    std::shared_ptr<System> m_system;

    bool m_initialized;

    std::unique_ptr<GlobalStepBehavior> m_global_step_behavior;
    AlgorithmData m_algorithm_data;
    mcl::MicroTimer m_micro_timer;

    double evaluate_auglag_objective(bool lagged_u) const;
};

}  // namespace wrapd

#endif  // SRC_ADMM_HPP_

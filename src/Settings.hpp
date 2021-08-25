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

#ifndef SRC_SETTINGS_HPP_
#define SRC_SETTINGS_HPP_

#include <string>

#include "Math.hpp"

namespace wrapd {

struct Settings {
    bool parse_args(int argc, char **argv);  // parse from terminal args. Returns true if help()
    void help();  // -help  print details, parse_args returns true if used

    //
    // IO Settings
    //
    class IO {
     public:
        IO() :
            m_input_mesh(""),
            m_verbose(1),
            m_should_save_data(true)
             {
        }

        std::string input_mesh() const { return m_input_mesh; }
        void input_mesh(std::string val) { m_input_mesh = val; }

        int verbose() const { return m_verbose; }
        void verbose(int val) { m_verbose = val; }

        bool should_save_data() const { return m_should_save_data; }
        void should_save_data(bool val) { m_should_save_data = val; }

     private:
        std::string m_input_mesh;
        int m_verbose;
        bool m_should_save_data;
    };
    IO m_io;

    int m_admm_iters;
    double m_gamma;  // Threshold for reweighting
    double m_earlyexit_tol;  // early exiting from the ADMM main loop
    bool m_earlyexit_asap;  // early exit from ADMM main loop as soon as there are no inversions

    class GStep {
     public:
        GStep() :
            m_max_iters(100),
            m_linesearch(LineSearch::None),
            m_kappa(2.0) {
        }

        enum class LineSearch {
            None,
            Backtracking,
        };

        int max_iters() const { return m_max_iters; }
        void max_iters(int val) { m_max_iters = val; }

        LineSearch linesearch() const { return m_linesearch; }
        void linesearch(LineSearch val) { m_linesearch = val; }

        double kappa() const { return m_kappa; }
        void kappa(double val) { m_kappa = val; }

     private:

        int m_max_iters;
        LineSearch m_linesearch;
        double m_kappa;
    };
    GStep m_gstep;

    enum class RotAwareness {
        ENABLED,
        DISABLED
    };

    enum class Reweighting {
        ENABLED,
        DISABLED
    };

    double beta_static() const { return m_beta_static; }
    void beta_static(double val) { m_beta_static = val; }

    double beta_min() const { return m_beta_min; }
    void beta_min(double val) { m_beta_min = val; }

    double beta_max() const { return m_beta_max; }
    void beta_max(double val) { m_beta_max = val; }

    bool weight_clamp_easing() const { return m_weight_clamp_easing; }
    void weight_clamp_easing(bool val) { m_weight_clamp_easing = val; }

    bool viewer() const { return m_viewer; }
    void viewer(bool val) { m_viewer = val; }

    RotAwareness m_rot_awareness;
    Reweighting m_reweighting;
    double m_beta_static;
    double m_beta_min;
    double m_beta_max;
    bool m_weight_clamp_easing;
    bool m_viewer;

    //
    Settings() {
        m_rot_awareness = RotAwareness::ENABLED;
        m_reweighting = Reweighting::ENABLED;
        m_admm_iters = 5000;
        m_gamma = 1.5;
        m_earlyexit_tol = 1.e-12;
        m_earlyexit_asap = false;
        m_beta_static = 1.0;
        m_beta_min = 1.0;
        m_beta_max = 1.e9;
        m_weight_clamp_easing = true;
        m_viewer = true;
    }
};
}  // namespace wrapd
#endif  // SRC_SETTINGS_HPP_

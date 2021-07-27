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

#include <string>
#include <sstream>
#include <iostream>

#include "Settings.hpp"

namespace wrapd {

bool Settings::parse_args(int argc, char **argv ) {
    std::vector<std::string> cmd_list;
    std::vector<std::stringstream> val_list;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if ((arg.at(0) == '-') && 
                (arg.at(1) != '0') &&
                (arg.at(1) != '1') &&
                (arg.at(1) != '2') &&
                (arg.at(1) != '3') &&
                (arg.at(1) != '4') &&
                (arg.at(1) != '5') &&
                (arg.at(1) != '6') &&
                (arg.at(1) != '7') &&
                (arg.at(1) != '8') &&
                (arg.at(1) != '9') &&
                (arg.at(1) != '.')) {
            cmd_list.push_back(arg);
            val_list.push_back(std::stringstream());
        } else {
            int val_list_size = val_list.size();
            val_list[val_list_size -1] << arg << " ";
        }
    }

    const int num_commands = cmd_list.size();
    for (int i = 0; i < num_commands; i++) {

        std::string arg = cmd_list[i];
        std::stringstream val;
        val << val_list[i].rdbuf();

        if ( arg == "-help" || arg == "--help" || arg == "-h" ) {
            help();
            return true;
        } else if (arg == "-input" || arg == "-i") { 
            std::string temp_string;
            val >> temp_string;
            m_io.input_mesh(temp_string);
        } else if (arg == "-p") {
            int temp_int;
            val >> temp_int;
            m_io.verbose(temp_int);
        } else if (arg == "-s") { 
            bool temp_bool;
            val >> temp_bool;
            m_io.should_save_data(temp_bool);
        } else if (arg == "-it") {
            val >> m_admm_iters;
        } else if (arg == "-global_it") { 
            int temp_val;
            val >> temp_val;
            m_gstep.max_iters(temp_val);
        } else if (arg == "-global_ls") {
            int temp_val;
            val >> temp_val;
            if (temp_val == 0) {
                m_gstep.linesearch(GStep::LineSearch::None);
            } else if (temp_val == 1) {
                m_gstep.linesearch(GStep::LineSearch::Backtracking);
            } else {
                throw std::runtime_error("Error: invalid value provided for argument -global_linesearch");
            }
        } else if (arg == "-kappa") {
            double temp_val;
            val >> temp_val;
            m_gstep.kappa(temp_val);
        } else if (arg == "-rotaware" || arg == "-ra") {
            int temp_val;
            val >> temp_val;
            if (temp_val == 0) {
                m_rot_awareness = RotAwareness::DISABLED;
            } else if (temp_val == 1) {
                m_rot_awareness = RotAwareness::ENABLED;
            } else {
                throw std::runtime_error("Error: Invalid value provided for argument -rotaware");
            }
        } else if (arg == "-reweighting" || arg == "-rw") {
            int temp_val;
            val >> temp_val;
            if (temp_val == 0) {
                m_reweighting = Reweighting::DISABLED;
            } else if (temp_val == 1) {
                m_reweighting = Reweighting::ENABLED;
            } else {
                throw std::runtime_error("Error: Invalid value provided for argument -reweighting");
            }
        } else if (arg == "-gamma") {
            double temp;
            val >> temp;
            m_gamma = temp;
        } else if (arg == "-tol") {
            val >> m_earlyexit_tol;
        } else if (arg == "-exit_asap") {
            val >> m_earlyexit_asap;
        } else if (arg == "-beta_static") { 
            double temp;
            val >> temp;
            beta_static(temp);
        } else if (arg == "-beta_min") {
            double temp;
            val >> temp;
            beta_min(temp);
        } else if (arg == "-beta_max") {
            double temp;
            val >> temp;
            beta_max(temp);
        } else if (arg == "-easing") {
            bool temp_val;
            val >> temp_val;
            weight_clamp_easing(temp_val);
        }
    }

    // Check if last arg is one of our no-param args
    std::string arg(argv[argc-1]);
    if ( arg == "-help" || arg == "--help" || arg == "-h" ) {
        help();
        return true;
    }

    return false;
}  // end parse settings args

void Settings::help() {
    std::stringstream ss;
    ss << "\n==========================================\nArgs:\n" <<
        " General: \n" <<
        " -i: input mesh (path to an .obj file) \t\t [REQUIRED]\n" <<
        " -p: print stats to terminal while running \t [Default: 1]       (0=disabled, 1=enabled) \n" <<
        " -s: save stats to file after running \t\t [Default: 1]       (0=disabled, 1=enabled) \n" <<
        " -it: maximum # ADMM iterations \t\t [Default: 5000]    (any integer > 0) \n" <<
        " -tol: tolerance for ADMM early exit \t\t [Default: 1.e-12]  (any small float > 0) \n" <<
        " -exit_asap: only use ADMM as an initializer \t [Default: 0]       (0=disabled, 1=enabled) \n" <<
        " -beta_static: weight^2 mult (no reweighting) \t [Default: 1.0]     (any float >= 1.0) \n" <<
        "\n" <<
        " Rotation awareness: \n" <<
        " -ra: rotation awareness toggle \t\t [Default: 1]       (0=disabled, 1=enabled) \n" <<
        " -global_it: max # global step L-BFGS iters \t [Default: 100]     (any integer > 0) \n" <<        
        " -global_ls: linesearch in the global step \t [Default: 0]       (0=none, 1=backtracking) \n" <<
        " -kappa: global step early exit coeff. \t\t [Default: 2.0]     (any float > 0)] \n" <<
        "\n" <<
        " Dynamic reweighting: \n" <<
        " -rw: dynamic reweighting toggle \t\t [Default: 1]       (0=disabled, 1=enabled) \n" <<
        " -gamma: threshold for reweighting \t\t [Default: 1.5]     (any float >= 1.0) \n" <<
        " -beta_min: smallest allowable weight^2 mult \t [Default: 1.0]     (any float >= 1.0) \n" <<
        " -beta_max: largest allowable weight^2 mult \t [Default: 1.e9]    (any float >= 1.0) \n" <<
        " -easing: gradually increase max weight^2 \t [Default: 1]       (0=disabled, 1=enabled) \n" <<
    "==========================================\n";
    printf("%s", ss.str().c_str() );
    exit(0);
}


}  // namespace wrapd

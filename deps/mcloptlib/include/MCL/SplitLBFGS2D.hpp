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
// Original Author: Ioannis Karamouzas
// Modified by George E. Brown (https://www-users.cse.umn.edu/~brow2327/)

#ifndef MCL_SPLITLBFGS2D_H
#define MCL_SPLITLBFGS2D_H

#include "SplitMinimizer.hpp"
#include "LinearSolver.hpp"

namespace mcl {
namespace optlib {

// L-BFGS implementation based on Nocedal & Wright Numerical Optimization book (Section 7.2)
// DIM = dimension of the problem
// M = history window
//
template<typename Scalar, int DIM, int M=6>
class SplitLBFGS2D : public SplitMinimizer2D<Scalar,DIM> {
private:
	typedef Eigen::Matrix<Scalar,DIM,M> MatM;
	typedef Eigen::Matrix<Scalar,M,1> VecM;
	typedef Eigen::Matrix<Scalar,DIM,2> MatX2;

	std::vector<MatX2> s;
	std::vector<MatX2> y;
	std::vector<double> alpha;
	std::vector<double> rho;

	MatX2 q;
	MatX2 grad;
	MatX2 grad_old;
	MatX2 x_old;
	MatX2 x_last;

	double m_tol;
	int m_k;

public:
	#if USE_PARDISO
		PardisoSolver m_pardiso_solver;
	#else
		LinearSolver m_linear_solver;
	#endif

	bool show_denom_warning; // Print out warning for zero denominators
	bool m_has_A0;
	double m_kappa;

	SplitLBFGS2D() : show_denom_warning(false) {
		this->m_settings.max_iters = 50;
		show_denom_warning = this->m_settings.verbose > 0 ? true : false;
		m_has_A0 = false;
		m_tol = 1.e-15;
		m_kappa = 2.0;
		m_k = 0;
	}

	void update_A0(Eigen::SparseMatrix<double, Eigen::RowMajor> &A0, bool first_time = true) {
		m_has_A0 = true;

		if (first_time) {
			#ifdef USE_PARDISO
				m_pardiso_solver.compute(A0);
			#else
				m_linear_solver.compute(A0);
			#endif
		} else {
			#ifdef USE_PARDISO
				m_pardiso_solver.factorize(A0);
			#else
				m_linear_solver.factorize(A0);
			#endif
		}
		
		m_k = 0;
	}

	void clear_history() {
		m_k = 0;
	}

	void update_max_iters(int max_it) {
		this->m_settings.max_iters = max_it;
	}

	// void update_earlyexit_coeff(double earlyexit_coeff) {
	//	m_earlyexit_coeff = earlyexit_coeff;
	// }

	void update_kappa(double kappa) {
		m_kappa = kappa;
	}

	void update_earlyexit_tol(double tol) {
		m_tol = tol;
	}

	// Returns number of iterations used
	int minimize(SplitProblem2D<Scalar,DIM> &problem, Eigen::MatrixX2d &x) {

		if(m_k == 0 && DIM==Eigen::Dynamic ){
			int dim = x.rows();
			s.resize(M);
			y.resize(M);
			alpha.resize(M);
			rho.resize(M);
			q = MatX2::Zero(dim, 2);
			grad = MatX2::Zero(dim, 2);
			grad_old = MatX2::Zero(dim, 2);
			x_old = MatX2::Zero(dim, 2);
			x_last = MatX2::Zero(dim, 2);
		}

		// if (m_k != 0 && (x - x_last).squaredNorm() < 1.e-12) {
			problem.gradient(grad);
		// } else {
		//	problem.gradient(x, grad);
		//}

		Scalar alpha_init = 1.0;

		int global_iter = 0;
		int max_iters = this->m_settings.max_iters;
		int verbose = this->m_settings.verbose;

		if ((grad - grad_old).cwiseProduct(grad - grad_old).sum() < 1.e-13) {
			return global_iter;
		} else {
			global_iter = 1;
		}

		bool breaking_out = false;
		for(; global_iter <= max_iters; global_iter++) {

			x_old = x;
			grad_old = grad;
			q = grad;

			// L-BFGS first - loop recursion		
			int iter = std::min(M, m_k);
			for(int i = iter - 1; i >= 0; --i){
				rho[i] = 1.0 / ((s[i].cwiseProduct(y[i])).sum());
				alpha[i] = rho[i] * (s[i].cwiseProduct(q)).sum();
				q = q - alpha[i] * y[i];
			}

			if (m_has_A0) {
				#ifdef USE_PARDISO
					for (int c = 0; c < 2; c++) {
						q.col(c) = m_pardiso_solver.solve(q.col(c));
					}
				#else
					for (int c = 0; c < 2; c++) {
						q.col(c) = m_linear_solver.solve(q.col(c));
					}
				#endif
			} else {
				throw std::runtime_error("Error: A0 is required but wasn't supplied.");
			}

			// L-BFGS second - loop recursion	
			for(int i = 0; i < iter; ++i){
				Scalar beta = rho[i] * (y[i].cwiseProduct(q)).sum();
				q = q + s[i] * (alpha[i] - beta);
			}

			// is there a descent
			Scalar dir = grad.cwiseProduct(q).sum();
			if (std::isnan(dir)) {
				std::cout << "Decrement is NaN. Exiting.\n";
				exit(0);
			}
			if (dir <= 0) {
				// std::cout << "Decrement is <= 0. It is " << dir << ". Exiting.\n";
				// exit(0);
			}
			if (dir <= 0 || std::isnan(dir)) {
				q = grad;
				max_iters -= m_k;
				m_k = 0;
				alpha_init = std::min(1.0, 1.0 / grad.template lpNorm<Eigen::Infinity>() );
			}

			Scalar rate = 1.0;
			if (m_tol > 1.e-12) {
				rate = this->linesearch(x, -q, problem, alpha_init);
			}
			if( rate <= 0 ){
				if( verbose > 0 ){ printf("LBFGS::minimize: Failure in linesearch\n"); }
				return SplitMinimizer2D<Scalar,DIM>::FAILURE;
			}

			x_last = x;
			x -= rate * q;

			problem.gradient(x,grad);

			MatX2 s_temp = x - x_old;
			MatX2 y_temp = grad - grad_old;
			
			// update the history
			if(m_k < M){
				s[m_k] = s_temp;
				y[m_k] = y_temp;
			}
			else {
				s.erase(s.begin());
				s.push_back(s_temp);
				y.erase(y.begin());
				y.push_back(y_temp);
			}

			// if (dir > 0 && (m_earlyexit_coeff * 0.5 * dir < m_tol 
			// 		|| m_earlyexit_coeff * 0.5 * dir < 1.e-12)) {
			if (dir > 0 && (dir < m_kappa * m_tol || dir < m_kappa * 1.e-12)) {
				m_k += 1;
				x_old = x;
				x_last = x;
				breaking_out = true;
				break;
			}

			Scalar denom = y_temp.cwiseProduct(y_temp).sum();
			if( std::abs(denom) <= 0 ){
				if( show_denom_warning ){
					printf("LBFGS::minimize Warning: Encountered a zero denominator\n");
				}
				breaking_out = true;
				break;
			}
			alpha_init = 1.0;

			m_k++;
		}
		m_k = std::min(M, m_k);

		if (breaking_out) {
			return global_iter;
		} else {
			return global_iter - 1;
		}

	} // end minimize
};


}
}

#endif

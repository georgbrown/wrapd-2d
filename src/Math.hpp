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

#ifndef SRC_MATH_HPP_
#define SRC_MATH_HPP_

#include <Eigen/SparseCholesky>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace wrapd {
namespace math {

using VecX = Eigen::VectorXd;
using VecXi = Eigen::VectorXi;
using MatX = Eigen::MatrixXd;
using MatXi = Eigen::MatrixXi;
using MatX2 = Eigen::MatrixX2d;
using Mat2X = Eigen::Matrix2Xd;
using SpMat = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using DiagMat = Eigen::DiagonalMatrix<double, Eigen::Dynamic>;
using RowVec3 = Eigen::RowVector3d;

using Triplet = Eigen::Triplet<double>;
using Triplets = std::vector<Triplet>;

using Doubles = std::vector<double>;

using Cholesky = Eigen::SimplicialLDLT< Eigen::SparseMatrix<double>, Eigen::Lower>;

using AlignedBox = Eigen::AlignedBox<double, 3>;

// Singular value decomposition
template <typename TYPE>
using JacobiSVD = Eigen::JacobiSVD<TYPE>;
static constexpr int ComputeFullU = Eigen::ComputeFullU;
static constexpr int ComputeFullV = Eigen::ComputeFullV;

// Mapping between vectors and matrices
template <typename TYPE>
using Map = Eigen::Map<TYPE>;

// Vectors of doubles
template <int DIM>
using Vec = Eigen::Matrix<double, DIM, 1>;
using Vec2 = Vec<2>;
using Vec3 = Vec<3>;
using Vec4 = Vec<4>;

// Vectors of floats
template <int DIM>
using Vecf = Eigen::Matrix<float, DIM, 1>;
using Vec2f = Vecf<2>;
using Vec3f = Vecf<3>;
using Vec4f = Vecf<4>;

// Vectors of integers
template <int DIM>
using Veci = Eigen::Matrix<int, DIM, 1>;
using Vec1i = Veci<1>;
using Vec2i = Veci<2>;
using Vec3i = Veci<3>;
using Vec4i = Veci<4>;

template <int ROWS, int COLS>
using Mat = Eigen::Matrix<double, ROWS, COLS>;

using Mat1x3 = Mat<1, 3>;
using Mat2x2 = Mat<2, 2>;
using Mat2x3 = Mat<2, 3>;
using Mat3x2 = Mat<3, 2>;
using Mat3x3 = Mat<3, 3>;
using Mat3x4 = Mat<3, 4>;

inline double clamp(const double minval, const double val, const double maxval) {
    double retval = val;
    if (val < minval) {
        retval = minval;
    }
    if (val > maxval) {
        retval = maxval;
    }
    return retval;
}

inline void svd(
        const math::Mat2x2& F,
        math::Vec2& sigma,
        math::Mat2x2& U,
        math::Mat2x2& V,
        bool prevent_flips = true) {
    math::JacobiSVD<math::Mat2x2> svd(F, math::ComputeFullU | math::ComputeFullV);
    sigma = svd.singularValues();
    U = svd.matrixU();
    V = svd.matrixV();

    if (prevent_flips && (U*V.transpose()).determinant() < 0.) {
        math::Mat2x2 J = math::Mat2x2::Identity();
        J(1, 1) = -1.0;
        if (U.determinant() < 0.) {
            U = U * J;
            sigma[1] = -sigma[1];
        }
        if (V.determinant() < 0.0) {
            math::Mat2x2 Vt = V.transpose();
            Vt = J * Vt;
            V = Vt.transpose();
            sigma[1] = -sigma[1];
        }
    }
}

inline void polar(
        const math::Mat2x2& F,
        math::Mat2x2& R,
        math::Mat2x2& S,
        bool prevent_flips = true) {
    math::Vec2 sigma;
    math::Mat2x2 U;
    math::Mat2x2 V;
    svd(F, sigma, U, V, prevent_flips);
    R = U * V.transpose();
    S = V * sigma.asDiagonal() * V.transpose();
}

inline math::Mat2x2 rot(const math::Mat2x2& A) {
    math::Vec2 b(A(0, 0) + A(1, 1), A(0, 1) - A(1, 0));
    b.normalize();
    math::Mat2x2 R2;
    R2(0, 0) = R2(1, 1) = b(0);
    R2(0, 1) = b(1);
    R2(1, 0) = -b(1);
    return R2;
}

inline math::Mat2x2 sym(const math::Mat2x2& A) {
    math::Mat2x2 R = rot(A);
    return R.transpose() * A;
}

}  // namespace math
} // namespace wrapd


#endif  // SRC_MATH_HPP_
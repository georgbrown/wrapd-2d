// These parameterization helper functions were obtained from the AKVF source code:
// https://github.com/sebastian-claici/AKVFParam
// 
// Written by Sebastian Claici and licensed under the GNU General Public License

#ifndef MCL_INITPARAM_HPP
#define MCL_INITPARAM_HPP 1

#include <igl/readOBJ.h>
#include <igl/boundary_loop.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/harmonic.h>
#include <igl/edges.h>
#include <igl/flipped_triangles.h>
#include <igl/is_vertex_manifold.h>
#include <igl/read_triangle_mesh.h>
#include <igl/vertex_components.h>
#include <igl/edge_topology.h>
#include <igl/is_edge_manifold.h>

namespace param_utils {

void map_vertices_to_circle_area_normalized(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const Eigen::VectorXi& bnd,
  Eigen::MatrixXd& UV) {
  
  Eigen::VectorXd dblArea_orig; // TODO: remove me later, waste of computations
  igl::doublearea(V,F, dblArea_orig);
  double area = dblArea_orig.sum()/2;
  double radius = sqrt(area / (M_PI));

  // Get sorted list of boundary vertices
  std::vector<int> interior,map_ij;
  map_ij.resize(V.rows());
  interior.reserve(V.rows()-bnd.size());

  std::vector<bool> isOnBnd(V.rows(),false);
  for (int i = 0; i < bnd.size(); i++)
  {
    isOnBnd[bnd[i]] = true;
    map_ij[bnd[i]] = i;
  }

  for (int i = 0; i < (int)isOnBnd.size(); i++)
  {
    if (!isOnBnd[i])
    {
      map_ij[i] = interior.size();
      interior.push_back(i);
    }
  }

  // Map boundary to unit circle
  std::vector<double> len(bnd.size());
  len[0] = 0.;

  for (int i = 1; i < bnd.size(); i++)
  {
    len[i] = len[i-1] + (V.row(bnd[i-1]) - V.row(bnd[i])).norm();
  }
  double total_len = len[len.size()-1] + (V.row(bnd[0]) - V.row(bnd[bnd.size()-1])).norm();

  UV.resize(bnd.size(),2);
  for (int i = 0; i < bnd.size(); i++)
  {
    double frac = len[i] * (2. * M_PI) / total_len;
    UV.row(map_ij[bnd[i]]) << radius*cos(frac), radius*sin(frac);
  }

}

inline void get_flips(const Eigen::MatrixXd& V,
               const Eigen::MatrixXi& F,
               const Eigen::MatrixXd& uv,
               std::vector<int>& flip_idx) {
  (void)(V);
  flip_idx.resize(0);
  for (int i = 0; i < F.rows(); i++) {

    Eigen::Vector2d v1_n = uv.row(F(i,0)); Eigen::Vector2d v2_n = uv.row(F(i,1)); Eigen::Vector2d v3_n = uv.row(F(i,2));

    Eigen::MatrixXd T2_Homo(3,3);
    T2_Homo.col(0) << v1_n(0),v1_n(1),1;
    T2_Homo.col(1) << v2_n(0),v2_n(1),1;
    T2_Homo.col(2) << v3_n(0),v3_n(1),1;
    double det = T2_Homo.determinant();
    assert (det == det);
    if (det < 0) {
      //cout << "flip at face #" << i << " det = " << T2_Homo.determinant() << endl;
      flip_idx.push_back(i);
    }
  }
}

inline int count_flips(const Eigen::MatrixXd& V,
              const Eigen::MatrixXi& F,
              const Eigen::MatrixXd& uv) {

  std::vector<int> flip_idx;
  get_flips(V,F,uv,flip_idx);

  
  return flip_idx.size();
}


int get_euler_char(const Eigen::MatrixXd& V,
              const Eigen::MatrixXi& F) {

  int euler_v = V.rows();
  Eigen::MatrixXi EV, FE, EF;
  igl::edge_topology(V, F, EV, FE, EF);
  int euler_e = EV.rows();
  int euler_f = F.rows();
    
  int euler_char = euler_v - euler_e + euler_f;
  return euler_char;
}

void check_mesh_for_issues(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::VectorXd& areas) {

  Eigen::SparseMatrix<double> A;
  igl::adjacency_matrix(F,A);

  Eigen::MatrixXi C, Ci;
  igl::vertex_components(A, C, Ci);
  //cout << "#Connected_Components = " << Ci.rows() << endl;
  //cout << "is edge manifold = " << igl::is_edge_manifold(V,F) << endl;
  int connected_components = Ci.rows();
  if (connected_components!=1) {
    std::cout << "Error! Input has multiple connected components" << std::endl; exit(1);
  }
  int euler_char = get_euler_char(V, F);
  if (!euler_char) {
    std::cout << "Error! Input does not have a disk topology, it's euler char is " << euler_char << std::endl; exit(1);
  }
  bool is_edge_manifold = igl::is_edge_manifold(F);
  if (!is_edge_manifold) {
    std::cout << "Error! Input is not an edge manifold" << std::endl; exit(1);
  }
  const double eps = 1e-14;
  for (int i = 0; i < areas.rows(); i++) {
    if (areas(i) < eps) {
      std::cout << "Error! Input has zero area faces" << std::endl; exit(1);
    }
  }
}

inline void dirichlet_on_circle(
		std::string filename, // obj file
		Eigen::MatrixXd &V3D, // 3D vertices
		Eigen::MatrixXd &VTC, // 2D UV tex init
		Eigen::MatrixXi &F) {

	typedef Eigen::Matrix<double,Eigen::Dynamic,1> VectorXS;
    Eigen::VectorXd areas;

	// std::cout << "\tReading mesh object" << std::endl;
	igl::read_triangle_mesh(filename, V3D, F);

	// set uv coords scale
	igl::doublearea(V3D, F, areas); areas /= 2.;

	check_mesh_for_issues(V3D, F, areas);
	// std::cout << "\tMesh is valid!" << std::endl;

	double mesh_area = areas.sum();

	V3D /= sqrt(mesh_area);

      // init (dirichlet)
      Eigen::VectorXi b;
      igl::boundary_loop(F,b);
      Eigen::MatrixXd bc;
      map_vertices_to_circle_area_normalized(V3D,F,b,bc);
      
      //igl::harmonic(V,F,bnd,bnd_uv,1,uv);

      Eigen::SparseMatrix<double> L,M,Mi;
      igl::cotmatrix(V3D,F,L);
      Eigen::SparseMatrix<double> Q = -L;
      VTC.resize(V3D.rows(),bc.cols());

      // Slow version (without Pardiso)
      const VectorXS B = VectorXS::Zero(V3D.rows(),1);
      igl::min_quad_with_fixed_data<double> data;
      igl::min_quad_with_fixed_precompute(Q,b,Eigen::SparseMatrix<double>(),true,data);

      for(int w = 0;w<bc.cols();w++) {
        const VectorXS bcw = bc.col(w);
        VectorXS Ww;
        if(!igl::min_quad_with_fixed_solve(data,B,bcw,VectorXS(),Ww)) return;
        VTC.col(w) = Ww;
      }
}

inline void tutte_on_circle(
		std::string filename, // obj file
		Eigen::MatrixXd &V3D, // 3D vertices
		Eigen::MatrixXd &VTC, // 2D UV tex init
		Eigen::MatrixXi &F) {
  using namespace Eigen;
  // typedef Matrix<double,Dynamic,1> VectorXS;
// generate boundary conditions to a circle

	// typedef Eigen::Matrix<double,Eigen::Dynamic,1> VectorXS;
    Eigen::VectorXd areas;

	// std::cout << "\tReading mesh object" << std::endl;
	igl::read_triangle_mesh(filename, V3D, F);

	// set uv coords scale
	igl::doublearea(V3D, F, areas); areas /= 2.;

	check_mesh_for_issues(V3D, F, areas);
	// std::cout << "\tMesh is valid!" << std::endl;

	double mesh_area = areas.sum();

	V3D /= sqrt(mesh_area);

  Eigen::SparseMatrix<double> A;
  igl::adjacency_matrix(F,A);

  Eigen::VectorXi b;
  igl::boundary_loop(F,b);
  Eigen::MatrixXd bc;
  map_vertices_to_circle_area_normalized(V3D,F,b,bc);

  
  // sum each row 
  Eigen::SparseVector<double> Asum;
  igl::sum(A,1,Asum);
  //Convert row sums into diagonal of sparse matrix
  Eigen::SparseMatrix<double> Adiag;
  igl::diag(Asum,Adiag);
  // Build uniform laplacian
  Eigen::SparseMatrix<double> Q;
  Q = Adiag - A;
  VTC.resize(V3D.rows(),bc.cols());


  // Slow version (without Pardiso)
  const Eigen::VectorXd B = Eigen::VectorXd::Zero(V3D.rows(),1);
  igl::min_quad_with_fixed_data<double> data;
  igl::min_quad_with_fixed_precompute(Q,b,Eigen::SparseMatrix<double>(),true,data);
  for(int w = 0;w<bc.cols();w++)
  {
    const Eigen::VectorXd bcw = bc.col(w);
    Eigen::VectorXd Ww;
    if(!igl::min_quad_with_fixed_solve(data,B,bcw,Eigen::VectorXd(),Ww))
    {
      return;
    }
    VTC.col(w) = Ww;
  }
}

}  // namespace param_utils

#endif

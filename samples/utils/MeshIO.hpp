// Copyright (c) 2017 University of Minnesota
// 
// MCLSCENE Uses the BSD 2-Clause License (http://www.opensource.org/licenses/BSD-2-Clause)
// Redistribution and use in source and binary forms, with or without modification, are
// permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of
//    conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list
//    of conditions and the following disclaimer in the documentation and/or other materials
//    provided with the distribution.
// THIS SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR  A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY OF MINNESOTA, DULUTH OR CONTRIBUTORS BE 
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
// OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
// IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
// OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// By Matt Overby (http://www.mattoverby.net)

#ifndef MCL_MESHIO_H
#define MCL_MESHIO_H

#include "TriangleMesh.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>

namespace mcl {
namespace meshio {

	static inline bool load_obj( TriangleMesh *mesh, std::string file, bool normals=false, bool texcoords=false, bool colors=false );

	static inline bool save_obj( const TriangleMesh *mesh, std::string file, bool normals=false, bool texcoords=false);


}; // namespace io


//
// Implementation
//

namespace io_helper {
	static std::string to_lower( std::string s ){
		std::transform( s.begin(), s.end(), s.begin(), ::tolower );
		return s;
	}
}

static inline bool meshio::load_obj( TriangleMesh *mesh, std::string file,
	bool normals, bool texcoords, bool colors ){

	mesh->clear();
	std::ifstream infile( file.c_str() );

	if( infile.is_open() ){

		std::string line;
		while( std::getline( infile, line ) ){

			std::stringstream ss(line);
			std::string tok; ss >> tok;
			tok = io_helper::to_lower(tok);

			if( tok == "v" ){ // Vertex
				double x, y, z; ss >> x >> y >> z; // vertices
				mesh->vertices.emplace_back( Vec3d(x,y,z) );
				if( colors ){
//					double cx, cy, cz; // colors
//					if( ss >> cx >> cy >> cz ){ mesh->colors.emplace_back( Vec3d(cx,cy,cz) ); }
				}
			}

			else if( tok == "vt" && texcoords ){ // Tex coord
				double u, v; ss >> u >> v;
				mesh->texcoords.emplace_back( Vec2d(u,v) );
			}

			else if( tok == "vn" && normals ){ // Normal
				double x, y, z; ss >> x >> y >> z; // vertices
				mesh->normals.emplace_back( Vec3d(x,y,z) );
			}

			else if( tok == "f" ){ // face
				Vec3i face;
				for( size_t i=0; i<3; ++i ){ // Get the three vertices
					std::vector<int> f_vals;
					{ // Split the string with the / delim
						std::string f_str; ss >> f_str;
						std::stringstream ss2(f_str); std::string s2;
						while( std::getline(ss2, s2, '/') ){
							if( s2.size()==0 ){ continue; }
							f_vals.emplace_back( std::stoi(s2)-1 );
						}
					}
					face[i] = f_vals.size() > 0 ? f_vals[0] : -1;
				}
				if( face[0]>=0 && face[1]>=0 && face[2]>=0 ){ mesh->faces.emplace_back(face); }
			} // end parse face

		} // end loop lines

	} // end load obj
	else { std::cerr << "\n**mcl::meshio::load_obj Error: Could not open file " << file << std::endl; return false; }

	// Double check our file
	if( mesh->texcoords.size() != mesh->vertices.size() && mesh->texcoords.size()>0 && texcoords ){
		std::cerr << "\n**mcl::meshio::load_obj Error: Failed to load texture coordinates." << std::endl;
		mesh->texcoords.clear();
		return false;
	}

	// Double check our file
	if( mesh->normals.size() != mesh->vertices.size() && mesh->normals.size()>0 && normals ){
		std::cerr << "\n**mcl::meshio::load_obj Warning: Normals should be per-vertex (removing them)." << std::endl;
		mesh->normals.clear();
		return false;
	}

	return true;

} // end load obj


static inline bool meshio::save_obj( const TriangleMesh *mesh, std::string filename, bool normals, bool texcoords){

	bool suppress_tex = !texcoords;
	bool suppress_normals = !normals;

	int fsize = filename.size();
	if( fsize<4 ){ printf("\n**TriangleMesh::save Error: Filetype must be .obj\n"); return false; }
	std::string ftype = filename.substr( fsize-4,4 );
	if( ftype != ".obj" ){ printf("\n**TriangleMesh::save Error: Filetype must be .obj\n"); return false; }

	std::ofstream fs;
	fs.open( filename.c_str() );
	fs << "# Generated with mclscene by Matt Overby (www.mattoverby.net)";

	int nv = mesh->vertices.size();
	int nn = mesh->normals.size();
	int nt = mesh->texcoords.size();
	if( suppress_tex ){ nt=0; }
	if( suppress_normals ){ nn=0; }

	for( int i=0; i<nv; ++i ){
		fs << "\nv " << std::scientific << std::setprecision(15) << mesh->vertices[i][0] << ' ' << mesh->vertices[i][1] << ' ' << mesh->vertices[i][2];
	}
	for( int i=0; i<nn; ++i ){
		fs << "\nvn " << std::scientific << std::setprecision(15) << mesh->normals[i][0] << ' ' << mesh->normals[i][1] << ' ' << mesh->normals[i][2];
	}
	for( int i=0; i<nt; ++i ){
		fs << "\nvt " << std::scientific << std::setprecision(15) << mesh->texcoords[i][0] << ' ' << mesh->texcoords[i][1];
	}
	std::setprecision(6);
	int nf = mesh->faces.size();
	for( int i=0; i<nf; ++i ){
		Vec3i f = mesh->faces[i]; f[0]+=1; f[1]+=1; f[2]+=1;
		if( nn==0 && nt==0 ){ // no normals or texcoords
			fs << "\nf " << f[0] << ' ' << f[1] << ' ' << f[2];
		} else if( nn==0 && nt>0 ){ // no normals, has texcoords
			fs << "\nf " << f[0]<<'/'<<f[0]<< ' ' << f[1]<<'/'<<f[1]<< ' ' << f[2]<<'/'<<f[2];
		} else if( nn>0 && nt==0 ){ // has normals, no texcoords
			fs << "\nf " << f[0]<<"//"<<f[0]<< ' ' << f[1]<<"//"<<f[1]<< ' ' << f[2]<<"//"<<f[2];
		} else if( nn>0 && nt>0 ){ // has normals and texcoords
			fs << "\nf " << f[0]<<'/'<<f[0]<<'/'<<f[0]<< ' ' << f[1]<<'/'<<f[1]<<'/'<<f[1]<< ' ' << f[2]<<'/'<<f[2]<<'/'<<f[2];
		}
	} // end loop faces

	fs << "\n";
	fs.close();
	return true;

} // end save obj

}; // namespace mcl

#endif
